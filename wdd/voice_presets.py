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

# ============================================================================
# 人声效果预设库
# 适用场景：播客、配音、直播、游戏、AI语音等
# 设计思路：通过均衡、动态、空间、染色四大维度塑造不同声景
# ============================================================================


def VoicePresetStudioClean():
    """
    播音室干声 - 纯净、贴近、无染

    链路逻辑：
    - 高通滤波：切除80Hz以下低频噪音（空调、麦克风振动）
    - 齿音控制：3kHz 轻微提升，增强清晰度而不刺耳
    - 动态稳定：-22dB阈值轻柔压缩，平滑音量波动
    - 增益补偿：+2dB 输出电平
    """
    studio_clean = Pedalboard(
        [
            HighpassFilter(80),  # 切除无用低频
            PeakFilter(3000, 3, 1.0),  # 临场感提升
            Compressor(threshold_db=-22, ratio=3),  # 轻柔动态控制
            Gain(3),  # 音量补偿
        ]
    )
    return studio_clean


def VoicePresetStageHost():
    """
    舞台主持 - 温暖、有力、适度空间

    链路逻辑：
    - 低频夯实：180Hz 轻微提升，增加厚度
    - 清晰定位：2.8kHz 适度提升，穿透背景噪音
    - 温暖染色：轻微失真模拟模拟设备饱和感
    - 空间包裹：0.35小房间混响，不喧宾夺主
    """
    stage_host = Pedalboard(
        [
            HighpassFilter(90),  # 切除低频杂讯
            PeakFilter(2800, 4, 0.9),  # 清晰度定位
            PeakFilter(180, 2, 0.8),  # 厚度补充
            Distortion(2),  # 模拟暖声染色
            Compressor(threshold_db=-18, ratio=4),  # 中等压缩
            Reverb(room_size=0.35, wet_level=0.25),  # 空间感
            Gain(3),  # 音量补偿
        ]
    )
    return stage_host


def VoicePresetMotivational():
    """
    激情演讲 - 高能量、冲刺感、张力拉满

    链路逻辑：
    - 低频冲击：160Hz 提升，增强气势
    - 高频锋芒：3kHz 强力提升，穿透力MAX
    - 失真染色：轻度饱和，增加兴奋度
    - 空间扩张：0.55房间+短延迟，营造集会声场
    """
    motivational = Pedalboard(
        [
            PeakFilter(160, 3, 0.8),  # 气势低频
            PeakFilter(3000, 5, 1.0),  # 穿透高频
            Distortion(3),  # 兴奋感染色
            Compressor(threshold_db=-20, ratio=5),  # 强压缩稳定能量
            Reverb(room_size=0.55, wet_level=0.3),  # 大厅感
            Delay(delay_seconds=0.12, feedback=0.2, mix=0.15),  # 回声支持
            Gain(4),  # 音量补偿
        ]
    )
    return motivational


def VoicePresetInnerVoice():
    """
    内心独白 - 梦幻、抽离、沉浸

    链路逻辑：
    - 空间主导：大房间混响，创造抽离感
    - 合唱调制：0.6Hz慢速合唱，声音分离度提升
    - 轻柔压缩：仅稳定动态，不破坏氛围
    - 无EQ干预：保留原始音色特质
    """
    inner_voice = Pedalboard(
        [
            Reverb(room_size=0.6, wet_level=0.4, damping=0.4),  # 沉浸空间
            Chorus(rate_hz=0.6, depth=0.3, mix=0.2),  # 声音分离
            Compressor(threshold_db=-25, ratio=3),  # 仅做稳定
        ]
    )
    return inner_voice


def VoicePresetRadio():
    """
    广播电台 - 复古窄带、压感、回忆杀

    链路逻辑：
    - 频带限制：120Hz~5kHz，模拟AM广播频宽
    - 过载染色：高失真度，复刻晶体管收音机质感
    - 极限压缩：高压缩比，制造压扁动态的年代感
    """
    radio = Pedalboard(
        [
            HighpassFilter(120),  # 切除低频，模拟小喇叭
            LowpassFilter(5000),  # 切除高频，模拟窄带
            Distortion(4),  # 晶体管过载感
            Compressor(threshold_db=-16, ratio=5),  # 压扁动态
            Gain(3),  # 音量补偿
        ]
    )
    return radio


def VoicePresetRadioVoice():
    """
    电台主播 - 温暖广播人声，介于广播与HiFi之间

    链路逻辑：
    - 近讲效应模拟：120Hz 提升，贴耳感
    - 空气感保留：3.2kHz提升，不失细节
    - 快速压缩：Attack 5ms，快速响应唇齿音
    - 微混响：0.3小空间，增加湿润度
    """
    radio_voice = Pedalboard(
        [
            HighpassFilter(70),  # 干净低频起点
            PeakFilter(120, 4, 0.9),  # 近讲效应模拟
            PeakFilter(3200, 5, 1.0),  # 清晰度保持
            Compressor(
                threshold_db=-18, ratio=4, attack_ms=5, release_ms=120
            ),  # 快速压缩响应
            Reverb(room_size=0.3, wet_level=0.15),  # 湿润感
        ]
    )
    return radio_voice


def VoicePresetCinematic():
    """
    影视旁白 - 沉厚、磁性、电影感

    链路逻辑：
    - 低频夯实：150Hz提升，男声磁性核心区
    - 细节定位：2.5kHz适度提升，对白清晰度
    - 轻柔压缩：低压缩比，保留动态细节
    - 空间定位：0.45房间，模拟录音棚声学环境
    """
    cinematic = Pedalboard(
        [
            PeakFilter(150, 2, 0.8),  # 磁性厚度
            PeakFilter(2500, 3, 1.0),  # 对白清晰度
            Compressor(threshold_db=-24, ratio=3),  # 保留动态
            Reverb(room_size=0.45, wet_level=0.25),  # 棚感
            Gain(3),  # 音量补偿
        ]
    )
    return cinematic


def VoicePresetMegaphone():
    """
    喇叭扩音 - 尖锐、远距离、户外感

    链路逻辑：
    - 极端频带：350Hz~3500Hz，模拟手持扩音器频响
    - 重度失真：6倍失真，喇叭过载特质
    - 极限压缩：Ratio 6，压扁所有动态差异
    """
    megaphone = Pedalboard(
        [
            HighpassFilter(350),  # 切除所有低频，尖锐感
            LowpassFilter(3500),  # 无高音，塑料感
            Distortion(6),  # 喇叭过载
            Compressor(threshold_db=-18, ratio=6),  # 极限压缩
            Gain(4),  # 音量补偿
        ]
    )
    return megaphone


def VoicePresetAIVoice():
    """
    AI机械语音 - 数字感、冷峻、合成特质

    链路逻辑：
    - 冷峻高频：3.5kHz强力提升，数字感
    - 合唱调制：2.0Hz快速合唱，产生数字分裂感
    - 无空间感：刻意不加混响，保持干冷
    """
    ai_voice = Pedalboard(
        [
            HighpassFilter(150),  # 切除温暖低频
            PeakFilter(3500, 5, 1.2),  # 数字锋锐感
            Chorus(rate_hz=2.0, depth=0.15, mix=0.25),  # 电子分裂
            Compressor(threshold_db=-20, ratio=4),  # 稳定电平
        ]
    )
    return ai_voice


def VoicePresetDreamVoice():
    """
    梦境空间 - 遥远、缥缈、非现实

    链路逻辑：
    - 大空间：0.8超大厅堂，完全抽离现实感
    - 合唱漂移：0.8Hz慢速合唱，声场漂浮
    - 长延迟：250ms延迟，创造回声层叠
    - 无压缩：保留所有动态起伏，梦境的无规则感
    """
    dream_voice = Pedalboard(
        [
            Reverb(room_size=0.8, wet_level=0.6),  # 超大厅堂
            Chorus(rate_hz=0.8, depth=0.4, mix=0.3),  # 空间漂移
            Delay(delay_seconds=0.25, feedback=0.4, mix=0.3),  # 回声层叠
        ]
    )
    return dream_voice


def VoicePresetGameNPC():
    """
    游戏NPC - 数字角色、非玩家、略怪异

    链路逻辑：
    - 锐化高频：2kHz强力提升+高Q值，角色辨识度
    - 轻微失真：2倍失真，数字生命特质
    - 合唱调制：1.5Hz合唱，非人类调制感
    """
    game_npc = Pedalboard(
        [
            PeakFilter(2000, 4, 1.1),  # 角色辨识频率
            Distortion(2),  # 电子质感
            Compressor(threshold_db=-22, ratio=4),  # 稳定音量
            Chorus(rate_hz=1.5, depth=0.25, mix=0.15),  # 调制感
        ]
    )
    return game_npc


def VoicePresetCapCutProVoice():
    """
    ✂️ 剪映专业人声 - 现代短视频标准器

    链路逻辑（五阶处理链）：

    ① 预处理清理
    - 70Hz高通：切除棚拍低频底噪
    - 6.5kHz陷波：精准抑制齿音s/z/c
    - 300Hz衰减：解除低频浑浊

    ② 语音塑形
    - 140Hz提升：贴麦厚度，解决手机录音发虚
    - 3.2kHz提升：抖音式清晰度，滑动屏幕区高频
    - 9kHz空气感：不刺耳的空气延伸

    ③ 动态控制
    - 前置轻压缩：8ms慢启动，保留瞬态
    - 主压缩：4ms快启动，稳定语句起伏

    ④ 密度增强
    - 2.5dB谐波：剪映式"磁性"秘密

    ⑤ 并行空间
    - 干声直达：保持口型清晰
    - 短混响：0.32房间，模拟AER模式
    - 微延迟：80ms，拓宽单声道录音

    ⑥ 最终整形
    - -1dB真峰值限制：适配所有平台响度标准
    - 3.5dB增益：达到主流LUFS
    """
    capcut_pro_voice = Pedalboard(
        [
            # =====================
            # 🎤 ① 预处理清理
            # =====================
            HighpassFilter(70),  # 去超低频
            PeakFilter(6500, -2, 1.2),  # 去刺耳齿音区
            PeakFilter(300, -2.5, 1.0),  # 去浑浊
            # =====================
            # 🎤 ② 语音塑形 EQ
            # =====================
            PeakFilter(140, 3.5, 0.9),  # 贴麦厚度
            PeakFilter(3200, 4.5, 1.1),  # 清晰存在感
            PeakFilter(9000, 2.5, 0.8),  # 空气感
            # =====================
            # 🎤 ③ 动态控制
            # （模拟多段压缩）
            # =====================
            Compressor(
                threshold_db=-24, ratio=2.5, attack_ms=8, release_ms=120
            ),  # 轻压缩控制整体
            Compressor(
                threshold_db=-18, ratio=4, attack_ms=4, release_ms=100
            ),  # 主压缩（语音稳定核心）
            # =====================
            # 🎤 ④ 密度增强（剪映秘密武器）
            # =====================
            Distortion(drive_db=2.5),
            # =====================
            # 🎤 ⑤ 并行空间设计
            # =====================
            Mix(
                [
                    Pedalboard([Gain(0)]),  # 干声
                    Pedalboard(
                        [
                            Reverb(
                                room_size=0.32, wet_level=0.18, damping=0.6, width=0.8
                            )
                        ]  # 短房间混响
                    ),
                    Pedalboard(
                        [Delay(delay_seconds=0.08, feedback=0.15, mix=0.12)]
                    ),  # 微延迟扩展声像
                ]
            ),
            # =====================
            # 🎤 ⑥ Loudness & 最终整形
            # =====================
            Limiter(threshold_db=-1.0),
            Gain(3.5),
        ]
    )
    return capcut_pro_voice


def VoicePresetVoiceSpeech():
    """
    现场演讲 - 大型场馆、感染力、远场投射

    相较于【激情演讲】更侧重空间投射：

    核心差异：
    - EQ策略：3.2kHz提升6dB，远场清晰度优先
    - 双压缩：第一层控动态，第二层提响度
    - 失真量：4dB较重染色，模拟PA过载
    - 延迟反馈：0.25反馈量，制造场馆回声层
    - 混响宽度：0.9立体声扩展，覆盖观众席
    """
    voice_speech = Pedalboard(
        [
            HighpassFilter(80),  # 切除低频风噪
            PeakFilter(3200, 6, 1.0),  # 远场投射核芯
            PeakFilter(160, 4, 0.8),  # 气势基底
            Compressor(-22, 3.5, 4, 110),  # 动态稳定层
            Compressor(-18, 5, 3, 100),  # 响度提升层
            Distortion(4),  # PA模拟染色
            Mix(
                [
                    Pedalboard([Gain(0)]),  # 直达声
                    Pedalboard(
                        [Reverb(room_size=0.45, wet_level=0.28, damping=0.6, width=0.9)]
                    ),  # 场馆混响
                    Pedalboard([Delay(0.12, 0.25, 0.18)]),  # 回声支持
                ]
            ),
            Limiter(-1),  # 防止过载
            Gain(4.5),  # 最终电平
        ]
    )
    return voice_speech
