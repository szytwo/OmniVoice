import os

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

import argparse
import gc
import json
import multiprocessing as mp
import time
import uuid
import warnings
from typing import Any, Dict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

from omnivoice import OmniVoiceGenerationConfig
from wdd.AudioProcessor import AudioProcessor
from wdd.file_utils import delete_old_files_and_folders, logging
from wdd.ModelManager import ModelManager
from wdd.TextProcessor import TextProcessor
from wdd.voice_effect import apply_preset


def get_main_args():
    parser = argparse.ArgumentParser(
        description="OmniVoice api server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose mode"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the api server on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the api server on"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--load_asr",
        dest="load_asr",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to load the ASR model (default: False).",
    )
    parser.add_argument(
        "--persistent_cuda",
        action="store_true",
        default=False,
        help="是否启用GPU常驻模式（启用后在同一进程内执行，适合单用户环境；不启用则每次生成都新建进程，适合多用户环境）",
    )
    parser.add_argument(
        "--max_retry",
        type=int,
        default=0,
        help="失败后的最大重试次数，0表示不重试",
    )

    # 设置显存比例限制（浮点类型，默认值为 0）
    parser.add_argument("--cuda_memory", type=float, default=0)

    return parser.parse_args()  # 每次调用都解析参数


argsMain = get_main_args()

result_input_dir = "results/input"
result_output_dir = "results/output"

audio_processor = AudioProcessor(result_input_dir, result_output_dir)


# 全局变量，显存常驻用
tts_global = None


def gpu_worker_init():
    logging.info("Initializing CUDA environment in persistent worker")
    init_cuda_env()


def set_seed(seed: int):
    """
    Set random seeds for reproducible generation.

    Args:
        seed: Random seed value (0 means no seed setting)
    """
    if seed != 0:
        # Clamp seed to valid NumPy range (0 to 2^32-1)
        clamped_seed = max(0, min(seed, 2**32 - 1))
        torch.manual_seed(clamped_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(clamped_seed)
        np.random.seed(clamped_seed)


def get_model_manager():
    """
    返回OmniVoice模型管理。
    persistent_cuda=True -> 返回全局常驻实例
    persistent_cuda=False -> 每次返回新实例
    """
    global tts_global

    if argsMain.persistent_cuda:
        if tts_global is None:
            tts_global = ModelManager(
                device=argsMain.device,
                dtype=argsMain.dtype,
                load_asr=argsMain.load_asr,
            )

        return tts_global
    else:
        return ModelManager(
            device=argsMain.device,
            dtype=argsMain.dtype,
            load_asr=argsMain.load_asr,
        )


def generate_voice_clone(
    text: str,
    prompt_text: str,
    prompt_wav: str,
    language: str = "Auto",
    speed: float = 1.00,
    seed: int = 0,
    instruct: str = "",
    **advanced_params,
):
    logging.info("Starting voice clone generation...")

    print(
        f"🔍 DEBUG: text={text}\n"
        f"🔍 DEBUG: prompt_text={prompt_text}\n"
        f"🔍 DEBUG: prompt_wav={prompt_wav}\n"
        f"🔍 DEBUG: language={language}\n"
        f"🔍 DEBUG: speed={speed}\n"
        f"🔍 DEBUG: instruct={instruct}\n"
        f"🔍 DEBUG: advanced_params={advanced_params}"
    )

    torch.cuda.synchronize()

    if not os.path.exists(result_output_dir):
        os.makedirs(result_output_dir)

    if not seed:
        seed = uuid.uuid4().int >> 96  # 基于 UUID，保证唯一性（几乎不会重复）

    logging.info(f"seed: {seed}")
    set_seed(seed)

    # 获取TTS实例
    model_manager = get_model_manager()
    model = model_manager.get_model("OmniVoice")
    sampling_rate = model.sampling_rate

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(advanced_params.get("num_step", 32)),
        guidance_scale=float(advanced_params.get("guidance_scale", 2.0)),
        denoise=bool(advanced_params.get("denoise", True)),
        preprocess_prompt=bool(advanced_params.get("preprocess_prompt", True)),
        postprocess_output=bool(advanced_params.get("postprocess_output", True)),
    )

    lang = language if (language and language != "Auto") else None
    prompt_text = prompt_text.strip() if prompt_text else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)

    duration = float(advanced_params.get("duration", 0.0))
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
        ref_audio=prompt_wav,
        ref_text=prompt_text,
    )

    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    wav_list = model.generate(**kw)

    # 指定保存文件的路径
    filename = f"{str(uuid.uuid4())}.wav"
    output_path = f"{result_output_dir}/{filename}"

    # 保存音频
    AudioProcessor.save_infer_audio(wav_list[0], sampling_rate, output_path)

    return output_path


def generate_voice_clone_timeout_with_retry(**kwargs):
    """
    包装 generate_voice_clone_timeout
    支持重试机制，避免偶发 CUDA invalid argument 错误
    超过重试次数直接抛异常
    """
    timeout_seconds = kwargs.get("timeout_seconds", 300)

    max_retry = max(argsMain.max_retry, 0)  # 最大重试次数，确保非负
    last_err = None

    for attempt in range(max_retry + 1):  # 包含第一次尝试 + 重试次数
        try:
            if argsMain.persistent_cuda:
                # GPU 常驻
                return generate_voice_clone(**kwargs)
            else:
                # 非常驻：每次新进程
                with ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=mp.get_context("spawn"),
                    initializer=gpu_worker_init,
                ) as executor:
                    future = executor.submit(generate_voice_clone, **kwargs)
                    return future.result(timeout=timeout_seconds)
        except Exception as e:
            last_err = e
            logging.error(
                f"[Retry {attempt}/{max_retry}] {type(e).__name__}: {str(e).splitlines()[0]}"
            )

            if argsMain.persistent_cuda:
                global tts_global
                tts_global = None

            time.sleep(5)  # 给 GPU 稍微恢复时间
        finally:
            if argsMain.persistent_cuda:
                clear_cuda_cache()

    # 超过重试次数仍失败，抛异常
    raise RuntimeError(
        f"generate_voice_clone failed after {max_retry} retries"
    ) from last_err


def generate_voice_clone_timeout(**kwargs):
    """
    包装 generate_voice_clone，增加超时机制与显存清理，防止卡死。
    """
    errcode = -1
    errmsg = "Unknown error"
    output_path = None
    gen_text = None

    text = kwargs.get("text", "")
    remove_silence = kwargs.get("remove_silence", False)
    min_silence = kwargs.get("min_silence", 300)
    keep_silence = kwargs.get("keep_silence", 200)

    # 记录开始时间
    start_time = time.time()

    # 文本预处理
    gen_text, lang = TextProcessor.clear_text(text)

    add_lang_tag = False  # 是否添加语言标签
    gen_text, lang = TextProcessor.ensure_sentence_ends_with_period(
        gen_text, add_lang_tag, lang
    )

    if lang in ["zh", "zh-cn"]:
        gen_text = TextProcessor.replace_chinese_number(gen_text)

    logging.info(f"{gen_text}")

    kwargs["text"] = gen_text

    try:
        output_path = generate_voice_clone_timeout_with_retry(**kwargs)
        # 短视频标准器
        output_path = apply_preset(output_path, "capcut_pro_voice")
        errcode = 0
        errmsg = "ok"
    except Exception as ex:
        TextProcessor.log_error(ex)

        errcode = -1
        errmsg = f"generate_voice_clone error: {str(ex)}"

        logging.error(errmsg)

    # 检查输出文件是否存在
    if errcode == 0 and output_path is not None and os.path.exists(output_path):
        if remove_silence:
            output_path = AudioProcessor.remove_silence_for_generated_wav(
                filename=output_path,
                min_silence=min_silence,
                keep_silence=keep_silence,
            )

        # 计算耗时
        elapsed = time.time() - start_time
        logging.info(f"Generation completed in {elapsed}.")
    else:
        errcode = -1
        errmsg = "Output file not generated."
        logging.error(f"{errmsg} Expected path: {output_path}")
        output_path = None

    # 删除过期文件
    delete_old_files_and_folders(result_output_dir, 1)
    delete_old_files_and_folders(result_input_dir, 1)

    return errcode, errmsg, output_path, gen_text


def init_cuda_env():
    if not torch.cuda.is_available():
        logging.warning("CUDA not available")
        return

    # 建议：只在这里设置
    if argsMain.cuda_memory > 0:
        logging.info(f"Set CUDA memory fraction: {argsMain.cuda_memory}")
        torch.cuda.set_per_process_memory_fraction(argsMain.cuda_memory)

    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.backends.cuda.enable_mem_efficient_sdp(False)


# 定义一个函数进行显存清理
def clear_cuda_cache():
    """
    清理PyTorch的显存和系统内存缓存。
    注意上下文，如果在异步执行，会导致清理不了
    """
    logging.info("Clearing GPU memory...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 GPU 完成所有 kernel

    # 强制进行垃圾回收
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 重置统计信息
        torch.cuda.reset_peak_memory_stats()

        log_gpu_memory("After clear cuda cache")


def log_gpu_memory(tag: str = ""):
    """
    打印当前 GPU 显存使用情况
    :param tag: 可选标记，用于区分位置
    """
    if not torch.cuda.is_available():
        logging.info(f"{tag} - CUDA not available")
        return
    # 打印显存日志
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()

    logging.info(
        f"[GPU Memory] {tag}:"
        f"\n>> Allocated: {allocated / 1024 ** 2:.2f} MB"
        f"\n>> Max Allocated: {max_allocated / 1024 ** 2:.2f} MB"
        f"\n>> Reserved: {reserved / 1024 ** 2:.2f} MB"
        f"\n>> Max Reserved: {max_reserved / 1024 ** 2:.2f} MB"
    )


# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。

app = FastAPI(title="OmniVoice API", version="v1.0")
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"],  # 允许跨域的headers，可以用来鉴别来源等作用。
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.get("/test")
async def test():
    """
    测试接口，用于验证服务是否正常运行。
    """
    return PlainTextResponse("success")


@app.post("/zero_shot/")
async def zero_shot(
    prompt_wav: UploadFile = File(
        ..., description="选择prompt音频文件，注意采样率不低于16khz"
    ),
    prompt_text: str = Form(
        "", description="请输入prompt文本，需与prompt音频内容一致，空为自动识别"
    ),
    text: str = Form(..., description="输入目标文本"),
    language: str = Form("Auto", description="输入目标语言"),
    advanced_params: str = Form(default="{}", description="高级参数(JSON字符串)"),
    spaker: float = Form(default=1.00, description="语速"),
    seed: int = Form(default=0, description="随机种子"),
    instruct: str = Form("", description="输入生成指令（可选）"),
    timeout_seconds: int = Form(default=600, description="生成超时时间，秒"),
    remove_silence: bool = Form(default=True, description="是否移除超出阈值的静音"),
    remove_silence_auto: bool = Form(
        default=True, description="按语速自动计算静音阈值"
    ),
    min_silence: int = Form(default=300, description="移除静音时长阈值，毫秒"),
    keep_silence: int = Form(default=200, description="保留静音时长，毫秒"),
    prompt_audio_fade: bool = Form(
        default=False, description="prompt音频是否淡入淡出处理"
    ),
    prompt_audio_fade_ms: int = Form(default=500, description="淡入淡出处理时长，毫秒"),
    prompt_audio_fade_clipping_protection: bool = Form(
        default=True,
        description="prompt音频淡入淡出后是否启用削波保护（防止处理后音频过大导致削波）",
    ),
    prompt_audio_fade_clipping_threshold: float = Form(
        default=0.95, description="削波保护阈值（0.0-1.0）"
    ),
):
    if remove_silence_auto:
        min_silence = int(min_silence / spaker)
        keep_silence = int(keep_silence / spaker)

    # 保存上传的音频文件
    prompt_audio_path = f"results/input/{prompt_wav.filename}"
    with open(prompt_audio_path, "wb") as buffer:
        buffer.write(prompt_wav.file.read())
    # 校验
    check = AudioProcessor.validate_audio(prompt_audio_path)

    if not check["ok"]:
        return JSONResponse({"errcode": -1, "errmsg": check["message"]})

    if prompt_audio_fade:
        prompt_audio_path = AudioProcessor.fade_in_out_with_path(
            prompt_audio_path,
            fade_ms=prompt_audio_fade_ms,
            clipping_protection=prompt_audio_fade_clipping_protection,
            clipping_threshold=prompt_audio_fade_clipping_threshold,
        )

    # 解析 advanced_params
    try:
        advanced_params_dict = json.loads(advanced_params)
        if not isinstance(advanced_params_dict, dict):
            raise ValueError
    except Exception:
        return JSONResponse(
            {"errcode": -1, "errmsg": "advanced_params 格式错误，必须是 JSON 对象"}
        )

    prompt_text = ""  # 目前接口层不处理 prompt_text，直接传空字符串，强制模型走 xvector_only 模式，避免偶发的 CUDA invalid argument 错误（推测与文本处理有关）
    # 调用原有的处理函数
    errcode, errmsg, wav_path, gen_text = generate_voice_clone_timeout(
        text=text,
        prompt_text=prompt_text,
        prompt_wav=prompt_audio_path,
        language=language,
        speed=spaker,
        seed=seed,
        instruct=instruct,
        timeout_seconds=timeout_seconds,
        remove_silence=remove_silence,
        min_silence=min_silence,
        keep_silence=keep_silence,
        **advanced_params_dict,
    )

    return JSONResponse(
        {
            "errcode": errcode,
            "errmsg": errmsg,
            "wav_path": wav_path,
            "gen_text": gen_text,
        }
    )


@app.get("/download")
async def download(
    wav_path: str = Query(..., description="输入wav文件路径"),
    name: str = Query(..., description="输入wav文件名"),
):
    """
    音频文件下载接口。
    """
    return FileResponse(
        path=wav_path, filename=name, media_type="application/octet-stream"
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)

        init_cuda_env()

        uvicorn.run(app, host=argsMain.host, port=argsMain.port)
    except Exception as mainEx:
        logging.error(mainEx)
        exit(0)
    finally:
        # 删除过期文件
        delete_old_files_and_folders(result_output_dir, 1)
        delete_old_files_and_folders(result_input_dir, 1)
        clear_cuda_cache()
