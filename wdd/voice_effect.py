import os

import pedalboard
from pedalboard import (
    Chorus,
    Compressor,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    Mix,
    PeakFilter,
    Pedalboard,
    Reverb,
)
from pedalboard.io import AudioFile

from wdd.file_utils import add_suffix_to_filename
from wdd.voice_presets import *

# 统一字典
VOICE_PRESETS = {
    "studio_clean": VoicePresetStudioClean,
    "stage_host": VoicePresetStageHost,
    "motivational": VoicePresetMotivational,
    "inner_voice": VoicePresetInnerVoice,
    "radio": VoicePresetRadio,
    "radio_voice": VoicePresetRadioVoice,
    "cinematic": VoicePresetCinematic,
    "megaphone": VoicePresetMegaphone,
    "ai_voice": VoicePresetAIVoice,
    "dream_voice": VoicePresetDreamVoice,
    "game_npc": VoicePresetGameNPC,
    "capcut_pro_voice": VoicePresetCapCutProVoice,
    "voice_speech": VoicePresetVoiceSpeech,
}


def apply_preset(input_file, preset_name, output_file=None):
    if output_file is None:
        output_file = add_suffix_to_filename(input_file, f"_{preset_name}")

    board = VOICE_PRESETS[preset_name]()

    # 读取音频
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    effected = board(audio, sr)
    effected = effected.T

    # 写入 WAV 32bit float
    with open(output_file, "wb") as f_out:
        with AudioFile(
            file_like=f_out,
            mode="w",
            samplerate=sr,
            num_channels=effected.shape[1],
            format="WAV",
            bit_depth=32,
        ) as f:
            f.write(effected)

    return output_file


if __name__ == "__main__":
    # 指定你的音频文件路径
    target_file = "./examples/wav/a.wav"

    if os.path.exists(target_file):
        # generate_preset_variants(target_file)
        apply_preset(target_file, "studio_clean")
        apply_preset(target_file, "stage_host")
        apply_preset(target_file, "motivational")
        apply_preset(target_file, "inner_voice")
        apply_preset(target_file, "radio")
        apply_preset(target_file, "cinematic")
        apply_preset(target_file, "megaphone")
        apply_preset(target_file, "ai_voice")
        apply_preset(target_file, "dream_voice")
        apply_preset(target_file, "game_npc")
        apply_preset(target_file, "radio_voice")
        apply_preset(target_file, "capcut_pro_voice")
        apply_preset(target_file, "voice_speech")
    else:
        print("错误：文件路径不存在，请检查路径设置。")
