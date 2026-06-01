import os
import uuid

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.effects import normalize

from wdd.file_utils import add_suffix_to_filename, logging


class AudioProcessor:
    def __init__(self, input_dir="results/input", output_dir="results/output"):
        """
        初始化音频处理器。
        :param input_dir: 输入文件目录
        :param output_dir: 输出文件目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def save_infer_audio(wav_data, sampling_rate, output_path=None):
        """
        将 infer() 返回的音频结果统一处理为 float32 单声道 tensor [1, samples]，
        并可选择保存为 float WAV 文件，保证 int16 -> float32 无失真。

        Args:
            wav_data (np.ndarray or torch.Tensor): 音频数据，形状为 (samples, channels) 或 (channels, samples)
            sampling_rate (int): 采样率
            output_path (str, optional): 保存 WAV 文件路径，默认不保存

        Returns:
            sampling_rate (int)
            audio (torch.FloatTensor): shape [1, samples], dtype float32, range [-1, 1]
        """
        # Debug: Check the actual values we're receiving
        print(f"🔍 DEBUG: wav_data dtype={wav_data.dtype}, shape={wav_data.shape}")
        print(f"🔍 DEBUG: wav_data min={wav_data.min()}, max={wav_data.max()}")

        # Convert numpy array (int16) to float32 following ComfyUI-IndexTTS2 reference
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.cpu().numpy()
        wav_data = np.asarray(wav_data)

        if wav_data.dtype == np.int16:
            wav_data = wav_data.astype(np.float32) / 32767.0
            print("np.int16 >> np.float32")
        elif wav_data.dtype != np.float32:
            wav_data = wav_data.astype(np.float32)
            print(f"{wav_data.dtype} >> np.float32")

        # Handle mono/stereo conversion matching reference implementation exactly
        mono = wav_data
        if mono.ndim == 2:
            if mono.shape[0] <= 8 and mono.shape[1] > mono.shape[0]:
                # Shape is (channels, samples) - average channels
                mono = mono.mean(axis=0)
            else:
                # Shape is (samples, channels) - average channels
                mono = mono.mean(axis=-1)
        elif mono.ndim > 2:
            mono = mono.reshape(-1, mono.shape[-1]).mean(axis=0)
        if mono.ndim != 1:
            mono = mono.flatten()

        # Convert to tensor [1, samples] format expected by our pipeline
        audio = torch.from_numpy(mono[None, :].astype(np.float32))

        print(f"🔍 DEBUG: final audio dtype={audio.dtype}, shape={audio.shape}")
        print(f"🔍 DEBUG: final audio min={audio.min():.6f}, max={audio.max():.6f}")

        # Save to WAV if requested
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            audio_np = audio.squeeze(0).cpu().numpy()
            # 覆盖原文件，保持 float32 格式
            sf.write(
                output_path, audio_np, sampling_rate, format="WAV", subtype="FLOAT"
            )
            print(">> wav file saved to:", output_path)

            return output_path
        else:
            return sampling_rate, wav_data

    # noinspection PyTypeChecker
    @staticmethod
    def remove_silence(audio, sr, top_db=30, silence_padding=0.2):
        """
        去除音频前后静音部分，并保留一定时长的静音。
        :param audio: np.ndarray，音频数据（Librosa 格式）。
        :param sr: int，采样率。
        :param top_db: int，静音检测阈值（分贝）。
        :param silence_padding: float，保留静音的时长（单位：秒）。默认保留 200ms。
        :return: np.ndarray，去除静音后的音频数据。
        """
        # 使用 librosa 检测静音并获取静音开始和结束的位置
        trimmed_audio, index = librosa.effects.trim(audio, top_db=top_db)
        # 计算要保留的静音样本数
        silence_samples = int(silence_padding * sr)
        # 保留前端的静音
        start_index = max(index[0] - silence_samples, 0)
        # 保留后端的静音
        end_index = min(index[1] + silence_samples, len(audio))

        logging.info(f"remove silence after: {start_index}-{end_index}")
        # 返回去除前后静音后，并且保留静音的音频
        return audio[start_index:end_index]

    @staticmethod
    def remove_silence_for_generated_wav(
        filename,
        min_silence=300,
        keep_silence=200,
        fade_duration=0.01,
        silence_thresh_db=50,
        frame_length=1024,
        hop_length=256,
    ):
        """
        方法说明：
        -----------
        - 使用 librosa 检测非静音区间；
        - 对每段非静音音频添加淡入淡出（可选）；
        - 在段间插入自定义长度的静音以替代长停顿；
        - 最终保存为 float32 精度的 WAV 文件（subtype='FLOAT'）。

        参数说明：
        ----------
        :param filename : str
            输入的音频文件路径（将会被处理并生成新文件）。
        :param min_silence : int
            静音最小间隔（毫秒）。如果两个非静音段之间的静音小于此长度，则保留原间隔；
            否则将间隔替换为固定的 keep_silence 长度。
        :param keep_silence : int
            静音替代长度（毫秒）。用于替代长静音，避免段间停顿太久。
        :param fade_duration : float
            每段非静音音频前后的淡入淡出时长（秒）。用于减少剪切导致的点击音或突兀感。
        :param silence_thresh_db : float
            静音检测阈值（单位 dB）。低于该阈值将被认为是静音，值越小越“敏感”。
        :param frame_length : int
            librosa 在检测静音时使用的帧长（单位：采样点数）。较大的值会更平滑但分辨率较低。
        :param hop_length : int
            librosa 在检测静音时滑动窗口的步长（单位：采样点数）。越小时间分辨率越高，但计算量也更大。

        返回：
        ------
        :return 处理后的新音频文件路径，文件名在原始文件基础上添加 "_silent" 后缀。
        """

        # 防止keep_silence > min_silence
        keep_silence = min_silence if keep_silence > min_silence else keep_silence
        # 读取音频（保持原采样率和 float32 精度）
        audio, sr = librosa.load(filename, sr=None, mono=True, dtype=np.float32)
        audio_len = len(audio)
        # 使用 librosa 分割非静音区段
        intervals = librosa.effects.split(
            audio,
            top_db=silence_thresh_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        intervals_len = len(intervals)

        logging.info(f"audio: {audio_len}, sr: {sr}, intervals: {intervals_len}")

        if intervals_len == 0:
            logging.info("音频中未检测到非静音段")
            return filename

        min_silence_len = int(sr * min_silence / 1000)
        keep_silence_len = int(sr * keep_silence / 1000)

        logging.info(
            f"min_silence_len: {min_silence_len}, keep_silence_len: {keep_silence_len}"
        )
        # 扩展每段区间前后，避免语音被切断
        padded_segments = []
        for i, (start, end) in enumerate(intervals):
            if i == 0:
                start = start - keep_silence_len if start > min_silence_len else 0

            orig_seg_ms = int((end - start) / sr * 1000)

            seg = audio[start:end]
            # 线性淡入／淡出：
            if fade_duration > 0:
                fade_len = int(sr * fade_duration)  # 10ms
                seg[:fade_len] *= np.linspace(0, 1, fade_len)
                seg[-fade_len:] *= np.linspace(1, 0, fade_len)

            padded_segments.append(seg)

            if i == intervals_len - 1:
                next_start = audio_len
            else:
                next_start = intervals[i + 1][0]
            # 间隔
            orig_silence_len = next_start - end
            orig_silence_ms = int(orig_silence_len / sr * 1000)

            # 如果停顿小于min_silence_len，保持，大于则按keep_silence
            if orig_silence_len > min_silence_len:
                pause_samples = keep_silence_len

                logging.info(
                    f"orig_seg: {i} {start}-{end} {orig_seg_ms}, "
                    f"orig_silence: {orig_silence_len} {orig_silence_ms}"
                )
            else:
                pause_samples = orig_silence_len
            # 段间插入静音
            silence_pad = np.zeros(pause_samples, dtype=np.float32)
            padded_segments.append(silence_pad)

        # 拼接所有非静音段
        final_audio = np.concatenate(padded_segments)
        silent_file = add_suffix_to_filename(filename, "_silent")
        # 覆盖原文件，保持 float32 格式
        sf.write(silent_file, final_audio, sr, format="WAV", subtype="FLOAT")

        return silent_file

    @staticmethod
    def volume_safely(
        audio: AudioSegment, volume_multiplier: float = 1.0
    ) -> AudioSegment:
        """
        安全地调整音频音量。
        :param audio: AudioSegment 对象，音频数据。
        :param volume_multiplier: float，音量倍数，1.0 为原音量，大于 1 提高音量，小于 1 降低音量。
        :return: 调整后的 AudioSegment 对象。
        """
        logging.info(f"volume_multiplier: {volume_multiplier}")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier 必须大于 0")
        # 计算增益（分贝），根据倍数调整
        gain_in_db = 20 * np.log10(volume_multiplier)
        # 应用增益调整音量
        audio = audio.apply_gain(gain_in_db)
        # 归一化到峰值以下
        audio = normalize(audio)

        return audio

    def generate_wav(self, audio_data, sample_rate, delay=0.0, volume_multiplier=1.0):
        """
        使用 pydub 将音频数据转换为 WAV 格式，并支持添加延迟。
        :param audio_data: numpy 数组，音频数据
        :param sample_rate: int，采样率
        :param delay: float，延迟时间（单位：秒），默认为 0
        :param volume_multiplier: float，音量倍数，默认为 1.0
        :return: 文件路径，生成的 WAV 文件路径
        """
        # 确保 audio_data 是 numpy 数组
        if not isinstance(audio_data, np.ndarray):
            raise ValueError("audio_data 必须是 numpy 数组。")
        # 生成静音数据（如果有延迟需求）
        if delay > 0:
            logging.info(f"生成延迟，delay: {delay}")
            num_silence_samples = int(delay * sample_rate)
            silence = np.zeros(num_silence_samples, dtype=audio_data.dtype)
            audio_data = np.concatenate((silence, audio_data), axis=0)
        # 检测音频数据类型并转换
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            logging.info(f"float32、float64 数据，量化到 int16")
            # audio_data = (audio_data * 32767).astype(np.int16)
            peak = np.max(np.abs(audio_data)) + 1e-8  # 避免除零
            audio_data = audio_data / peak * 0.99  # 留出1%的headroom
            audio_data = (audio_data * 32767).astype(np.int16)
            sample_width = 2  # 16-bit (2 bytes per sample)
        elif audio_data.dtype == np.int16:
            sample_width = 2  # 16-bit (2 bytes per sample)
        elif audio_data.dtype == np.int8:
            audio_data = audio_data.astype(np.int16) * 256  # 转换为 int16
            sample_width = 2  # 16-bit
        else:
            raise ValueError("audio_data.dtype 不正确。")
        # 检测声道数
        if len(audio_data.shape) == 1:  # 单声道
            channels = 1
        elif len(audio_data.shape) == 2:  # 多声道
            channels = audio_data.shape[1]
        else:
            raise ValueError(
                "audio_data.shape 格式不正确，必须是 1D 或 2D numpy 数组。"
            )
        # 使用 pydub 生成音频段
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )
        if volume_multiplier != 1.0:
            # 安全地增加音量
            audio_segment = self.volume_safely(audio_segment, volume_multiplier)
        else:
            # 归一化到峰值以下
            audio_segment = normalize(audio_segment)
        # 指定保存文件的路径
        filename = f"{str(uuid.uuid4())}.wav"
        wav_path = os.path.join(self.output_dir, filename)
        # 如果文件已存在，先删除
        if os.path.exists(wav_path):
            os.remove(wav_path)
        # 导出 WAV 文件
        audio_segment.export(wav_path, format="wav")

        return wav_path

    @staticmethod
    def audio_to_np_array(audio: AudioSegment):
        """将 AudioSegment 转换为 NumPy 数组"""
        return np.array(audio.get_array_of_samples())

    @staticmethod
    def np_array_to_audio(np_array, audio: AudioSegment):
        """将 NumPy 数组转换回 AudioSegment"""
        return AudioSegment(
            np_array.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels,
        )

    @staticmethod
    def validate_audio(file_path: str, min_sr: int = 8000) -> dict:
        """
        校验音频文件有效性
        :param file_path: 音频文件路径
        :param min_sr: 最低采样率
        :return: dict {"ok": bool, "message": str, "sample_rate": int, "duration": float}
        """
        samplerate: int = 0
        duration: float = 0

        try:
            audio = AudioSegment.from_file(file_path)
            samplerate = audio.frame_rate
            duration = len(audio) / 1000.0  # ms -> s

            if samplerate < min_sr:
                return {
                    "ok": False,
                    "message": f"采样率过低: {samplerate}Hz，需要 >= {min_sr}Hz",
                    "sample_rate": samplerate,
                    "duration": duration,
                }
            elif duration <= 0:
                return {
                    "ok": False,
                    "message": "音频无内容",
                    "sample_rate": samplerate,
                    "duration": duration,
                }

            return {
                "ok": True,
                "message": "音频有效",
                "sample_rate": samplerate,
                "duration": duration,
            }
        except Exception as e:
            return {
                "ok": False,
                "message": f"无效音频: {str(e)}",
                "sample_rate": 0,
                "duration": 0,
            }

    @staticmethod
    def clipping_protection(audio, threshold=0.95):
        """
        简单的削波保护
        threshold: 安全阈值，通常0.95-0.99
        """
        max_amp = np.max(np.abs(audio))

        if max_amp > threshold:
            # 等比缩小到threshold
            audio_protected = audio * (threshold / max_amp)
            logging.info(f"clipping protection: {max_amp:.3f} -> {threshold}")
            return audio_protected
        else:
            return audio

    @staticmethod
    def fade_in_out_with_path(
        audio_path: str,
        fade_ms: int = 30,
        clipping_protection: bool = True,
        clipping_threshold: float = 0.95,
    ) -> str:
        """
        对录音文件应用高质量的淡入淡出处理，消除起始和结束时的爆音/咔嗒声

        Args:
            audio_path (str): 音频文件路径
            fade_ms (int, optional): 淡入淡出时间（毫秒），默认值: 30
            clipping_protection (bool, optional): 是否启用削波保护，防止淡入淡出后音频过大导致削波，默认值: True
            clipping_threshold (float, optional): 削波保护阈值（0.0-1.0），默认值: 0.95（保留5%余量）

        Returns:
            str: 处理后的音频文件路径。处理失败时返回原始文件路径

        Raises:
            ValueError: 音频文件为空
            Exception: 文件读取、处理或保存过程中出现错误

        Example:
            >>> from audio_processor import AudioProcessor
            >>> result_path = AudioProcessor.fade_in_out_with_path(
            ...     "input.wav",
            ...     fade_ms=30,
            ...     clipping_protection=True
            ... )
            >>> print(f"处理后的文件: {result_path}")

        Note:
            1. 使用高质量的正弦平方根渐变曲线，避免高频伪影
            2. 音频过短时自动调整淡入淡出长度
            3. 启用削波保护可防止处理后的音频超出动态范围
            4. 默认保存为32位浮点WAV格式，保持最高质量
        """
        try:
            # 读取音频（保持原采样率，转换为单声道，使用float32高质量精度）
            audio, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)

            # 检查音频长度是否足够
            total_samples = len(audio)
            if total_samples == 0:
                raise ValueError("音频文件为空")

            # 计算淡入淡出长度（样本数）
            fade_len = int(sr * fade_ms / 1000.0)

            # 动态调整淡入淡出长度
            if total_samples < 2 * fade_len:
                # 音频太短，减少淡入淡出时间
                fade_len = total_samples // 4  # 使用1/4长度
                if fade_len < 2:  # 如果还是太短，至少保留2个样本
                    fade_len = min(2, total_samples)

            if fade_len == 0:
                # 如果淡入淡出长度为0，直接返回原文件
                return audio_path

            # 使用高质量的正弦平方根渐变曲线（最适合录音）
            # 这种曲线在0和1处的导数都是0，避免了高频伪影
            n = np.arange(fade_len, dtype=np.float32)

            # 正弦平方根曲线（sin(x)的平方根）
            fade_in = np.sin(n * np.pi / (2 * fade_len)) ** 0.5
            fade_out = np.sin((n + fade_len) * np.pi / (2 * fade_len)) ** 0.5

            # 应用淡入淡出（创建副本，避免修改原始数据）
            audio_faded = audio.copy()
            audio_faded[:fade_len] *= fade_in
            audio_faded[-fade_len:] *= fade_out

            if clipping_protection:
                # 轻微归一化，防止削波（只对淡入淡出处理后的音频）
                audio_faded = AudioProcessor.clipping_protection(
                    audio_faded, threshold=clipping_threshold
                )

            # 保存处理后的音频
            fade_file = add_suffix_to_filename(audio_path, "_fade")
            sf.write(fade_file, audio_faded, sr, format="WAV", subtype="FLOAT")

            # 输出处理信息
            duration = total_samples / sr

            logging.info(
                f"fade in out success:\n"
                f"   - audio_path: {audio_path}, fade_file: {fade_file}\n"
                f"   - duration: {duration:.2f}s, sr: {sr}Hz, fade ms: {fade_len/sr*1000:.1f}ms"
            )

            return fade_file
        except Exception as e:
            logging.error(f"fade in out error: {e}")

            # 返回原文件路径作为降级方案
            return audio_path

    @staticmethod
    def fade_in_out(wav, sampling_rate=22050, fade_ms=50):
        """
        对音频波形应用淡入淡出效果，避免起始和结束时的爆破音/咔嗒声

        Args:
            wav (np.ndarray): 音频波形数据，形状为 [通道数, 时间序列长度]
            sampling_rate (int, optional): 采样率，默认值: 22050
            fade_ms (int, optional): 淡入淡出时长（毫秒），默认值: 50

        Returns:
            np.ndarray: 经过淡入淡出处理的波形数据

        Raises:
            ValueError: 如果波形数据维度不正确

        Example:
            >>> processor = AudioProcessor()
            >>> processed = processor.fade_in_out(wav, sampling_rate=22050, fade_ms=50)

        Note:
            1. 使用汉宁窗实现平滑的淡入淡出效果
            2. 当音频长度小于2倍淡入淡出长度时，会自动调整淡入淡出长度
            3. 如果淡入淡出长度为0，则直接返回原波形
        """

        wav = np.asarray(wav, dtype=np.float32)

        # 计算淡入淡出长度
        fade_len = int(sampling_rate * fade_ms / 1000.0)
        wav_len = wav.shape[-1]

        # 过短处理（保持原逻辑）
        if wav_len < 2 * fade_len:
            fade_len = wav_len // 2

        if fade_len == 0:
            return wav

        # hann window（完全等价 torch.hann_window）
        fade_win = np.hanning(2 * fade_len).astype(np.float32)

        fade_in = fade_win[:fade_len]
        fade_out = fade_win[fade_len:]

        # 保持 broadcasting 行为一致
        wav[..., :fade_len] *= fade_in
        wav[..., -fade_len:] *= fade_out

        return wav
