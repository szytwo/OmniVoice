import datetime
import json
import os
import re
import traceback

import cn2an
import fasttext

from wdd.file_utils import logging


class TextProcessor:
    """
    文本处理工具类，提供多种文本相关功能。
    """

    @staticmethod
    def get_end_punctuations():
        # 定义哪些标点作为换行符号
        end_punctuations = [
            "，",
            "。",
            "！",
            "？",
            "；",
            "：",
            ",",
            ".",
            "!",
            "?",
            ";",
            ":",
        ]
        return end_punctuations

    @staticmethod
    def is_probably_chinese(text):
        contains_hanzi = bool(re.search(r"[\u4e00-\u9fff]", text))
        contains_japanese_kana = bool(re.search(r"[\u3040-\u30ff]", text))  # 日文假名
        contains_korean_hangul = bool(re.search(r"[\uac00-\ud7af]", text))  # 韩文音节
        return contains_hanzi and not (contains_japanese_kana or contains_korean_hangul)

    @staticmethod
    def clear_text(text, language=None):
        # 替换连续多个空格为一个空格
        text = re.sub(r"[ \t]+", " ", text)

        if not language:
            language = TextProcessor.detect_language(text)

        if language in ["zh", "zh-cn"]:
            text = TextProcessor.add_comma_before_newline(text, "，")
            text = TextProcessor.replace_corner_mark(text)
        else:
            text = TextProcessor.add_comma_before_newline(text, ",")

        text = text.replace("｜", "，")
        text = text.replace("|", ",")
        # 处理特殊数字。
        text = TextProcessor.normalize_digits(text)
        # 替换数字之间的及一些连接符号为空格或删除
        text = re.sub(r"(?<=\d)[,](?=\d)", "", text)
        text = TextProcessor.remove_bracket(text)
        # 最后替换连续多个空格为一个空格
        text = re.sub(r"[ \t]+", " ", text)
        # 清理连续标点，只保留最后一个
        text = re.sub(r"[，。！？；:;,.!?]+", lambda m: m.group(0)[-1], text)

        return text, language

    # noinspection PyTypeChecker
    @staticmethod
    def normalize_digits(text):
        """
        处理特殊数字。
        """
        digit_map = {
            # 上标数字
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            # 下标数字
            "₀": "0",
            "₁": "1",
            "₂": "2",
            "₃": "3",
            "₄": "4",
            "₅": "5",
            "₆": "6",
            "₇": "7",
            "₈": "8",
            "₉": "9",
            # 圈数字（只处理1–10）
            "①": "1",
            "②": "2",
            "③": "3",
            "④": "4",
            "⑤": "5",
            "⑥": "6",
            "⑦": "7",
            "⑧": "8",
            "⑨": "9",
            "⑩": "10",
        }

        result = []
        for c in text:
            code = ord(c)
            # 数学粗体 𝟎~𝟗 (U+1D7CE ~ U+1D7D7)
            if 0x1D7CE <= code <= 0x1D7D7:
                result.append(str(code - 0x1D7CE))
            # 数学等宽 𝟶~𝟿 (U+1D7F6 ~ U+1D7FF)
            elif 0x1D7F6 <= code <= 0x1D7FF:
                result.append(str(code - 0x1D7F6))
            # 全角数字 ０～９ (U+FF10 ~ U+FF19)
            elif 0xFF10 <= code <= 0xFF19:
                result.append(str(code - 0xFF10))
            # 特殊替换字典（上标、下标、圈数字等）
            elif c in digit_map:
                result.append(digit_map[c])
            else:
                result.append(c)

        return "".join(result)

    # noinspection PyTypeChecker
    @staticmethod
    def add_comma_before_newline(text: str, comma: str = "，") -> str:
        """
        在换行符（\n 或 \\N）前自动补充标点（默认逗号），然后去掉换行符。
        """

        # noinspection PyTypeChecker
        def needs_punctuation(segment: str) -> bool:
            """判断段落末尾是否需要补充标点"""
            return segment and segment[-1] not in TextProcessor.get_end_punctuations()

        segments = re.split(r"(\n|\\N)", text)  # 先按换行符拆分，并保留换行符
        cleaned_segments = []

        for i in range(len(segments)):
            if segments[i] in {"\n", "\\N"}:
                continue  # 直接跳过换行符，不添加到结果中
            if i < len(segments) - 1 and segments[i + 1] in {"\n", "\\N"}:
                # 如果后面是换行符，则检查是否需要补标点
                if needs_punctuation(segments[i]):
                    segments[i] += comma
            cleaned_segments.append(segments[i])

        return "".join(cleaned_segments)

    # remove meaningless symbol
    @staticmethod
    def remove_bracket(text):
        text = text.replace("（", "").replace("）", "")
        text = text.replace("【", "").replace("】", "")
        text = text.replace("`", "").replace("`", "")
        text = text.replace("——", " ")
        return text

    # replace special symbol
    @staticmethod
    def replace_corner_mark(text):
        text = text.replace("㎡", "平方米")
        text = text.replace("²", "平方")
        text = text.replace("³", "立方")
        return text

    @staticmethod
    def detect_language(text):
        """
        检测输入文本的语言。
        :param text: 输入文本
        :return: 返回检测到的语言代码（如 'en', 'zh', 'ja', 'ko'）
        """
        if TextProcessor.is_probably_chinese(text):
            return "zh"
        # 加载预训练的语言检测模型
        fasttext_model = fasttext.load_model(
            "./third_party/fastText/models/lid.176.bin"
        )

        try:
            lang = "zh"
            text = text.strip().replace("\n", "")
            if text:
                predictions = fasttext_model.predict(text, k=1)  # 获取 top-1 语言预测
                lang = predictions[0][0].replace("__label__", "")  # 解析语言代码
                # 判断是否有置信度值，并且置信度足够高
                # if len(predictions) > 1 and predictions[1]:
                #    confidence = predictions[1][0]  # 置信度
                #    lang = lang if confidence >= 0.5 else 'zh'

            logging.info(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logging.error(f"Language detection failed: {e}")
            return "zh"

    @staticmethod
    def ensure_sentence_ends_with_period(text, add_lang_tag: bool = False, lang=None):
        """
        确保输入文本以适当的句号结尾。
        :param text: 输入文本
        :param add_lang_tag: 是否添加语言标签
        :param lang: 语言
        :return: 修改后的文本
        """
        if not text.strip():
            return text, None  # 空文本直接返回
        # 根据文本内容添加适当的句号
        if not lang:
            lang = TextProcessor.detect_language(text)

        lang_tag = ""
        if add_lang_tag:
            if lang in ["zh", "zh-cn"]:  # 中文文本
                lang_tag = "<|zh|>"
            elif lang == "en":  # 英语
                lang_tag = "<|en|>"
            elif lang == "ja":  # 日语
                lang_tag = "<|jp|>"
            elif lang == "ko":  # 韩语
                lang_tag = "<|ko|>"
        # 判断是否已经以句号结尾
        if text[-1] in [".", "。", "！", "!", "？", "?"]:
            return f"{lang_tag}{text}", lang
        # 根据文本内容添加适当的句号
        if lang in ["zh", "zh-cn", "ja"]:  # 中文文本
            return f"{lang_tag}{text}。", lang
        else:  # 英文或其他
            return f"{lang_tag}{text}.", lang

    @staticmethod
    def log_error(exception: Exception, log_dir="error"):
        """
        记录错误信息到指定目录，并按日期时间命名文件。

        :param exception: 捕获的异常对象
        :param log_dir: 错误日志存储的目录，默认为 'error'
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        # 获取当前时间戳，格式化为 YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 创建日志文件路径
        log_file_path = os.path.join(log_dir, f"error_{timestamp}.log")
        # 使用 traceback 模块获取详细的错误信息
        error_traceback = traceback.format_exc()
        # 写入错误信息到文件
        with open(log_file_path, "w") as log_file:
            log_file.write(f"错误发生时间: {timestamp}\n")
            log_file.write(f"错误信息: {str(exception)}\n")
            log_file.write("堆栈信息:\n")
            log_file.write(error_traceback + "\n")

        logging.error(
            f"错误信息: {str(exception)}\n" f"详细信息已保存至: {log_file_path}"
        )

    @staticmethod
    def get_keywords(config_file="./custom/keywords.json"):
        with open(config_file, "r", encoding="utf-8") as file:
            words_list = json.load(file)
        return words_list

    # noinspection PyTypeChecker
    @staticmethod
    def add_quotation_mark(text, keywords, min_length=2):
        """
        在文本中为指定的词语添加引号，跳过长度小于 min_length 的词语。

        :param text: 输入文本
        :param keywords: 需要添加括号的词语列表
        :param min_length: 跳过添加括号的最小词语长度，默认为 2
        :return: 处理后的文本
        """

        text = text.replace("\n", "")
        text = TextProcessor.replace_blank(text)
        text = TextProcessor.replace_bracket(text)
        text = TextProcessor.replace_corner_mark(text)
        # logging.info(f'add quotation original text: {text}')

        # 常见引号标点符号
        punctuation = r"[\[\]（）【】《》““””‘’]"
        # 分割文本为引号内外的部分
        split_pattern = r"(“.*?”)"  # 非贪婪匹配引号内的内容
        # 按关键词长度从长到短排序
        keywords = sorted(keywords, key=len, reverse=True)

        for word in keywords:
            if len(word) >= min_length:
                parts = re.split(split_pattern, text)
                for i in range(len(parts)):
                    # 处理引号外的部分（偶数索引）
                    if i % 2 == 0:
                        current_part = parts[i]
                        # 匹配时确保目标词前后没有标点符号，且没有已有的引号
                        pattern = rf"(?<!“)(?<!{punctuation}){re.escape(word)}(?!{punctuation})(?<!”)"
                        # 使用正则表达式替换
                        current_part = re.sub(
                            pattern, f"“{word}”", current_part, flags=re.IGNORECASE
                        )
                        parts[i] = current_part
                # 合并所有部分
                text = "".join(parts)

        # logging.info(f'add quotation out text: {text}')

        return text

    # noinspection PyTypeChecker
    @staticmethod
    def add_dun_mark(text, keywords, min_length=2):
        """
        在文本中为指定的词语添加顿号，跳过长度小于 min_length 的词语。

        :param text: 输入文本
        :param keywords: 需要添加顿号的词语列表
        :param min_length: 跳过添加顿号的最小词语长度，默认为 2
        :return: 处理后的文本
        """

        text = text.replace("\n", "")
        text = TextProcessor.replace_blank(text)
        text = TextProcessor.replace_bracket(text)
        text = TextProcessor.replace_corner_mark(text)
        # logging.info(f'add quotation original text: {text}')

        # 常见引号标点符号
        punctuation = r"[\[\]（）【】《》““””‘’]"
        # 分割文本为顿号内外的部分
        split_pattern = r"(、.*?、)"  # 非贪婪匹配顿号内的内容
        # 按关键词长度从长到短排序
        keywords = sorted(keywords, key=len, reverse=True)

        for word in keywords:
            if len(word) >= min_length:
                parts = re.split(split_pattern, text)
                for i in range(len(parts)):
                    # 处理顿号外的部分（偶数索引）
                    if i % 2 == 0:
                        current_part = parts[i]
                        # 匹配时确保目标词前后没有标点符号，且没有已有的顿号
                        pattern = rf"(?<!、)(?<!{punctuation}){re.escape(word)}(?!{punctuation})(?<!、)"
                        # 使用正则表达式替换
                        current_part = re.sub(
                            pattern, f"、{word}、", current_part, flags=re.IGNORECASE
                        )
                        parts[i] = current_part
                # 合并所有部分
                text = "".join(parts)

        # logging.info(f'add quotation out text: {text}')

        return text

    # replace meaningless symbol
    @staticmethod
    def replace_bracket(text):
        text = text.replace("（", "“").replace("）", "”")
        text = text.replace("【", "“").replace("】", "”")
        return text

    # remove blank between chinese character
    # noinspection PyTypeChecker
    @staticmethod
    def replace_blank(text: str):
        out_str = []
        for i, c in enumerate(text):
            if c == " ":
                if (text[i + 1].isascii() and text[i + 1] != " ") and (
                    text[i - 1].isascii() and text[i - 1] != " "
                ):
                    out_str.append(c)
            else:
                out_str.append(c)
        return "".join(out_str)

    @staticmethod
    def smart_an2cn(
        input_str,
        mode="low",
        prefix="",
        suffix="",
        repl_single_two=False,
        repl_before_unit=True,
        unit_chars="千百万亿兆",
    ):
        """
        更灵活的智能阿拉伯数字转中文
        :param input_str: 输入数字（str/int/float）
        :param mode: 'low' 小写（默认） 或 'up' 大写
        :param prefix: 前缀（默认空）
        :param suffix: 后缀（默认空）
        :param repl_single_two: 是否替换孤立的 '二' 为 '两'
        :param repl_before_unit: 是否替换单位前的 '二' 为 '两'
        :param unit_chars: 单位字符列表（默认千百万亿兆）
        :return: 中文数字字符串
        """
        text = cn2an.an2cn(input_str, mode=mode)

        if mode == "low":
            if "点" in text:
                int_part, frac_part = text.split("点", 1)

                if repl_single_two and int_part == "二":
                    int_part = "两"
                elif repl_before_unit:
                    int_part = re.sub(rf"二(?=[{unit_chars}])", "两", int_part)

                text = int_part + "点" + frac_part
            else:
                if repl_single_two and text == "二":
                    text = "两"
                elif repl_before_unit:
                    text = re.sub(rf"二(?=[{unit_chars}])", "两", text)

        return f"{prefix}{text}{suffix}"

    @staticmethod
    def convert_datetime_to_chinese(datetime_str):
        parts = datetime_str.split(" ")
        date_part = parts[0]
        if "-" in date_part:
            year, month, day = date_part.split("-")
        elif "/" in date_part:
            year, month, day = date_part.split("/")
        else:
            return datetime_str

        time_parts = []
        if len(parts) > 1:
            time_part = parts[1]
            if "," in time_part:
                hms, millisecond = time_part.split(",")
                time_parts.extend(hms.split(":"))
                time_parts.append(millisecond)
            else:
                time_parts.extend(time_part.split(":"))

        def convert(num):
            return cn2an.an2cn(num.lstrip("0") or "0")

        chinese_parts = [
            f"{cn2an.an2cn(year, mode='direct')}年",
            f"{convert(month)}月",
            f"{convert(day)}日",
        ]

        if time_parts:
            time_labels = ["时", "分", "秒"]
            for i, part in enumerate(time_parts[:3]):
                chinese_parts.append(f"{convert(part)}{time_labels[i]}")

            if len(time_parts) > 3:
                chinese_parts.append(f"{convert(time_parts[3])}毫秒")

        return "".join(chinese_parts)

    @staticmethod
    def convert_num_to_chinese(input_str, suffix_rules=None):
        """
        根据输入字符串智能转换数字部分。
        :param input_str: 输入字符串（如 "2003计划"、"20年"、"2008份"）。
        :param suffix_rules: {"年":{"mode": "low"}}
        :return: 转换后的中文读法。
        """
        if suffix_rules is None:
            suffix_rules = {}
        if not input_str:
            return input_str
        input_str = input_str.replace(" ", "")
        prefix = ""
        if input_str.startswith("-") or input_str.startswith("+"):
            if suffix_rules:
                prefix = "减" if input_str.startswith("-") else "加"
            else:
                prefix = "负" if input_str.startswith("-") else "正"
            input_str = input_str.replace("+", "").replace("-", "")

        input_str = input_str.upper()
        # 检查是否有后缀
        for suffix in sorted(suffix_rules.keys(), key=len, reverse=True):
            rule = suffix_rules[suffix]
            suffix = suffix.upper()
            if input_str.upper().endswith(suffix):
                num_part = input_str[: -len(suffix)]  # 去掉后缀
                if "lengths" in rule and len(num_part) not in rule["lengths"]:
                    # 如果长度不符合规则，按普通数字转换
                    return TextProcessor.smart_an2cn(
                        num_part, prefix=prefix, suffix=suffix, repl_single_two=True
                    )
                else:
                    # 按规则中的模式转换
                    return TextProcessor.smart_an2cn(
                        num_part,
                        mode=rule["mode"],
                        prefix=prefix,
                        suffix=suffix,
                        repl_single_two=True,
                    )
        # 处理没有后缀年份范围（当前年份）
        if input_str.isdigit() and len(input_str) == 4:
            year = int(input_str)
            current_year = datetime.datetime.now().year
            if current_year - 1 <= year <= current_year + 1:
                return TextProcessor.smart_an2cn(input_str, mode="direct")
        # 如果没有后缀
        if input_str.isdigit() and input_str.startswith("0"):
            return TextProcessor.smart_an2cn(input_str, mode="direct")
        # 其他情况按普通数字转换
        return TextProcessor.smart_an2cn(input_str, prefix=prefix)

    @staticmethod
    def convert_time_to_chinese(time_str):
        """
        将时间字符串转换为中文读法。
        :param time_str: 时间字符串（如 "8:00"）。
        :return: 转换后的中文读法。
        """
        hours, minutes = map(int, time_str.split(":"))
        chinese_hours = cn2an.an2cn(str(hours), mode="low")
        chinese_minutes = cn2an.an2cn(str(minutes), mode="low")
        if minutes == 0:
            return f"{chinese_hours}点"
        else:
            return f"{chinese_hours}点{chinese_minutes}"

    @staticmethod
    def convert_timefull_to_chinese(time_str):
        """
        将时间字符串（如"8:00"）转换为中文读法（如"八点"）。
        :param time_str: 时间字符串。
        :return: 转换后的中文时间读法。
        """
        start, end = time_str.split("-")

        start_time = TextProcessor.convert_time_to_chinese(start)
        end_time = TextProcessor.convert_time_to_chinese(end)

        return f"{start_time}到{end_time}"

    # noinspection PyTypeChecker
    @staticmethod
    def replace_chinese_number(text):
        """
        将文本中的数字和单位替换为中文读法。
        :param text: 输入文本。
        :return: 替换后的文本。
        """
        # 逐字符转换的单位
        direct_units = ["年", "后"]
        # 普通数字转换的单位
        low_units = [
            # 时间类
            "季度",
            "月",
            "周",
            "日",
            "天",
            "小时",
            "分钟",
            "分",
            "秒",
            "毫秒",
            # 人数 / 次数
            "人",
            "位",
            "个",
            "名",
            "次",
            "遍",
            "轮",
            # 金额类
            "元",
            "角",
            "分",
            "人民币",
            "美元",
            "欧元",
            "日元",
            # 长度
            "千米",
            "公里",
            "米",
            "厘米",
            "毫米",
            "微米",
            "英里",
            "英寸",
            "尺",
            "丈",
            # 重量
            "吨",
            "千克",
            "公斤",
            "克",
            "毫克",
            "斤",
            "两",
            "盎司",
            "磅",
            "KG",
            # 容积、面积、温度
            "升",
            "毫升",
            "立方",
            "加仑",
            "平方",
            "亩",
            "公顷",
            "度",
            # 商品/量词
            "件",
            "份",
            "瓶",
            "盒",
            "包",
            "张",
            "本",
            "套",
            "台",
            "辆",
            "棵",
            "头",
            "块",
            "根",
            "支",
            "条",
            "双",
            "颗",
            "道",
            "款",
            "家",
            # 其他
            "多",
            "倍",
            "级",
            "档",
            "马力",
            "瓦",
            "千瓦",
        ]
        # 动态生成 suffix_rules
        suffix_rules = {}
        for unit in direct_units:
            if unit == "年":
                suffix_rules[unit] = {"lengths": [4, 4], "mode": "direct"}  # 逐字符转换
            else:
                suffix_rules[unit] = {"lengths": [2, 2], "mode": "direct"}
        for unit in low_units:
            suffix_rules[unit] = {"mode": "low"}  # 普通数字转换
        # 构建单位正则表达式
        units_pattern = "|".join(
            direct_units + low_units
        )  # 正则表达式匹配数字部分（包括带单位和不带单位的情况）
        # 匹配多种日期时间格式，2025-04-14 22:58:46,965等
        datetime_pattern = re.compile(
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}(?:\s\d{1,2}:\d{1,2}(?::\d{1,2}(?:,\d{1,3})?)?)?"
        )
        # 匹配时间格式，8:00-23:00
        timefull_pattern = re.compile(r"\d{1,2}:\d{2}-\d{1,2}:\d{2}")
        # 匹配时间格式，8:00
        time_pattern = re.compile(r"\d{1,2}:\d{2}")
        # 匹配电话号码，包括 +86-13987654321 或 010-1234567
        phone_pattern = re.compile(
            r"(?:\+?86-?1[3-9]\d{9}|1[3-9]\d{9}|\d{3,4}-\d{7,8}|\+\d{1,4}-\d{6,14})"
        )
        # 带单位
        bandunit_pattern = re.compile(rf"[+-]?\d+(?:\.\d+)?(?:\s*(?:{units_pattern}))")
        # 数学运算上下文映射
        math_op_map = {"+": "加", "-": "减", "*": "乘", "/": "除", "=": "等于"}
        math_pattern = re.compile(
            r"([+-]?\d+(?:\.\d+)?(?:\s*[+\-*/=]\s*[+-]?\d+(?:\.\d+)?)+)"
        )
        # 独立符号映射
        symbol_map = {"+": "加", "=": "等于", "@": "艾特"}
        # 带符号数字（正负）映射
        sign_pattern = re.compile(r"(?<!\d)(?P<sign>[+\-])(?P<num>\d+(?:\.\d+)?)(?!\d)")
        # 百分、千分、万分
        rate_pattern = re.compile(r"(?<!\d)([+-]?\d+(?:\.\d+)?)([%‰‱])")
        # 温度如：30℃、-10.5℉、+0.8℃、100 ℉ 等
        temperature_pattern = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(℃|℉)")
        # 版本号，IP
        v_ip_pattern = re.compile(r"\d+(?:\.\d+){2,}")
        # 其他数字
        number_pattern = re.compile(rf"(?<!\d)[+-]?\d+(?:\.\d+)?")

        def repl_math(m):
            expr = m.group(1).replace(" ", "")
            tokens = re.findall(
                r"[+\-*/=]{2}\d+(?:\.\d+)?|[+\-*/=]|\d+(?:\.\d+)?", expr
            )
            result = ""
            for token in tokens:
                if token in math_op_map:
                    result += math_op_map[token]
                    continue

                prefix = ""
                num = token
                if len(token) > 2 and token[:2] in {"+-", "--", "*-", "/-", "=-"}:
                    prefix = math_op_map[token[0]] + ("负" if token[1] == "-" else "")
                    num = token[2:]
                elif token.startswith("+") or token.startswith("-"):
                    prefix = "正" if token.startswith("+") else "负"
                    num = token[1:]

                result += TextProcessor.smart_an2cn(num, prefix=prefix)

            return result

        def repl_rate(m):
            num = m.group(1)
            rate = m.group(2)
            prefix = (
                "负" if num.startswith("-") else ("正" if num.startswith("+") else "")
            )
            num_clean = num.lstrip("+-")
            rate_map = {"%": "百分之", "‰": "千分之", "‱": "万分之"}
            return TextProcessor.smart_an2cn(
                num_clean, prefix=f"{prefix}{rate_map[rate]}"
            )

        def repl_temperature(match):
            num = match.group(1)
            temperature = match.group(2)
            prefix = ""

            if num.startswith("+"):
                prefix = "正"
                num = num[1:]
            elif num.startswith("-"):
                prefix = "零下"
                num = num[1:]

            suffix = "摄氏度" if temperature == "℃" else "华氏度"
            return TextProcessor.smart_an2cn(num, prefix=prefix, suffix=suffix)

        def repl_sign(m):
            sign, num = m.group("sign"), m.group("num")
            prefix = "正" if sign == "+" else "负"
            return TextProcessor.smart_an2cn(num, prefix=prefix)

        def repl_phone(m):
            s = m.group(0)
            s = s.replace("+", "").replace("-", "")
            return TextProcessor.smart_an2cn(s, mode="direct")

        def repl_datetime(m):
            s = m.group(0)
            return TextProcessor.convert_datetime_to_chinese(s)

        def repl_timefull(m):
            s = m.group(0)
            return TextProcessor.convert_timefull_to_chinese(s)

        def repl_time(m):
            s = m.group(0)
            return TextProcessor.convert_time_to_chinese(s)

        def repl_v_ip(m):
            parts = m.group().split(".")
            result = "点".join(
                "".join(cn2an.an2cn(d, mode="low") for d in part) for part in parts
            )
            return result

        def repl_bandunit(m):
            s = m.group(0)
            return TextProcessor.convert_num_to_chinese(s, suffix_rules)

        def repl_number(m):
            s = m.group(0)
            return TextProcessor.convert_num_to_chinese(s)

        try:
            # 1. 日期、时间
            text = datetime_pattern.sub(repl_datetime, text)
            text = timefull_pattern.sub(repl_timefull, text)
            text = time_pattern.sub(repl_time, text)
            # 2. 电话号码（优先于数学运算）
            text = phone_pattern.sub(repl_phone, text)
            # 3. 数学运算表达式
            text = math_pattern.sub(repl_math, text)
            # 4. 带单位数字、温度
            text = bandunit_pattern.sub(repl_bandunit, text)
            text = temperature_pattern.sub(repl_temperature, text)
            # 5. 百分、千分、万分、正负数字
            text = rate_pattern.sub(repl_rate, text)
            text = sign_pattern.sub(repl_sign, text)
            # 6. 版本号、IP、其他数字
            text = v_ip_pattern.sub(repl_v_ip, text)
            text = number_pattern.sub(repl_number, text)
            # 7. 剩余符号映射
            for sym, name in symbol_map.items():
                text = text.replace(sym, name)
        except Exception as e:
            TextProcessor.log_error(e)
            logging.error(f"replace chinese number error： {text}\n{str(e)}")

        return text

    @staticmethod
    def replace_pronunciation(text, keywords):
        """替换文本中的发音错误字"""
        for wrong_char, correct_char in keywords.items():
            text = text.replace(wrong_char, correct_char)

        return text

    @staticmethod
    def split_text(text: str, max_chars: int = 200):
        """
        将长文本尽量按照语义和标点进行切分。

        优先级：
        1. 句号、问号、感叹号、分号
        2. 逗号、顿号
        3. 空格
        4. 最终才强制按 max_chars 切
        """

        text = text.strip()

        if not text:
            return []

        if len(text) <= max_chars:
            return [text]

        AR_SPLIT_PUNCT = "،؟؛"  # 阿拉伯语\维吾尔语标点符号
        SPLIT_PUNCT = "：，；。！？,;!?.:" + AR_SPLIT_PUNCT
        # 先按照句末标点拆
        sentences = re.split(rf"(?<=[{SPLIT_PUNCT}])", text)

        result = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(current) + len(sentence) <= max_chars:
                current += sentence
            else:
                if current:
                    result.append(current)

                current = ""

        if current:
            result.append(current)

        return [x.strip() for x in result if x.strip()]
