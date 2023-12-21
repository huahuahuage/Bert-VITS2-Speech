import re
from typing import Tuple


def __split_jp_text(text: str):
    """
    提取日语
    文心一言说过：
    在日语中，连续出现的中文字符的数量通常是受限于句子的长度和语境。在正常的日语表达中，连续出现的中文字符一般不会超过两个。
    这是因为日语中的汉字通常只用于表示具有特定含义的词，而平假名和片假名则用于表示日语中的音节。因此，连续出现三个或更多的中文字符在日语中并不常见，也不是日语的常规用法。
    当然，在一些特定的语境下，比如使用汉字表示某种特殊的含义或者是因为某种特定的文化背景，可能会出现连续出现三个或更多的中文字符的情况。但这种情况并不常见，也不是日语的常规用法。

    所以这里匹配连续三个中文字符
    """
    jp_segments = []
    # 仅提取最多3个连续字符的中文字符
    zh_pattern = re.compile(r"[\u4e00-\u9fff]+")
    zh_char_dict = {}
    for match in zh_pattern.finditer(text):
        match_text = match.group().strip()
        if not match_text:
            continue
        # 仅提取最多3个连续字符
        if len(match_text) > 3:
            continue
        zh_char_dict[str(match.end())] = match_text

    # 提取日文
    jp_pattern = re.compile(
        r"[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F0-9\s]+"
    )  # 日文字符的Unicode范围

    jp_chars_list = []
    for match in jp_pattern.finditer(text):
        match_text = match.group().strip()
        if not match_text:
            continue
        char_start = match.start()
        char_end = match.end()
        # 是否有相邻中文字符，如果有，标记为日语
        zh_char = zh_char_dict.get(str(char_start), None)
        if zh_char:
            char_start = char_start - len(zh_char)
            match_text = zh_char + match_text
        jp_chars_list.append((char_start, char_end, match_text))

    if len(jp_chars_list) == 0:
        return []

    jp_segments = []
    font_jp_chars_tuple = jp_chars_list[0]
    font_start = font_jp_chars_tuple[0]
    font_end = font_jp_chars_tuple[1]
    font_text = font_jp_chars_tuple[2]

    for index, jp_chars_tuple in enumerate(jp_chars_list):
        if index == 0:
            continue
        start, end, text = jp_chars_tuple
        if start == font_end:
            # 如果当前元素的起始位置与前一个元素的结束位置相同，则将文本拼接到前一个元素的文本后面
            font_end = end
            font_text = font_text + text
        else:
            # 将前一个加入列表
            jp_segments.append((font_start, font_end, font_text, "JP"))
            # 否则，将当前元素加入前一个
            font_start = start
            font_end = end
            font_text = text

    # 将最后一个加入列表
    jp_segments.append((font_start, font_end, font_text, "JP"))
    # 打印结果

    return jp_segments


def __split_jp_en_text(text: str):
    """
    切割混合语言文本（日、英）
    """
    segments = []
    # 提取包含连续英文数字空格的字符串
    en_pattern = re.compile(r"[a-zA-Z0-9\s]+")
    for match in en_pattern.finditer(text):
        match_text = match.group().strip()
        if not match_text:
            continue
        segments.append((match.start(), match.end(), match_text, "EN"))

    # 提取日语
    jp_segments = __split_jp_text(text)
    segments = segments + jp_segments

    return segments


def __extract_chinese(text: str):
    """
    提取中文词和数字
    """
    # 提取中文词
    pattern = re.compile(r"[\u4e00-\u9fff0-9]+")
    chinese_chars = re.findall(pattern, text)
    if len(chinese_chars) == 0:
        return None
    return "，".join(chinese_chars)


def __divide_text(text: str, intervals: list):
    """
    三语言混合
    """
    # 对输入的区间列表按照起始索引进行排序
    intervals.sort(key=lambda x: x[0])
    # 判断输入的索引是否合法
    parts = []
    if len(intervals) > 0:
        if intervals[0][0] < 0 or intervals[-1][1] > len(text):
            return []

        # 划分出第一个区间前面的部分，并将其添加到列表头部
        first_start = intervals[0][0]

        if first_start > 0:
            first_part_text = __extract_chinese(text[:first_start])
            if first_part_text:
                first_part_info = (0, first_start, first_part_text, "ZH")
                parts.insert(0, first_part_info)

        # 划分出各个区间之间的部分，并存储到列表中
        for i in range(len(intervals) - 1):
            parts.append(intervals[i])
            start, end = intervals[i][1], intervals[i + 1][0]
            part_text = __extract_chinese(text[start:end])
            if not part_text:
                continue
            part_info = (intervals[i][1], intervals[i + 1][0], part_text, "ZH")
            parts.append(part_info)

        # 划分出最后一个区间后面的部分，并将其添加到列表末尾
        parts.append(intervals[-1])
        last_end = intervals[-1][1]
        # 如果text不是以最后的区间结尾
        if last_end < len(text) - 1:
            # 继续提取中文
            last_part_text = __extract_chinese(text[last_end:])
            if last_part_text:
                last_part_info = (last_end, len(text) - 1, last_part_text, "ZH")
                parts.append(last_part_info)
    else:
        zh_text = __extract_chinese(text)
        if zh_text:
            parts = [(0, len(zh_text), zh_text, "ZH")]

    # 返回划分后的结果
    return parts


def split_text(text: str) -> Tuple[list, list]:
    """
    自动切割混合文本
    """
    other_text_segments = __split_jp_en_text(text)
    text_segments = __divide_text(text, other_text_segments)

    if not text_segments:
        return None

    text_list = []
    language_list = []
    for text_segment in text_segments:
        text_list.append(text_segment[2])
        language_list.append(text_segment[3])
    return text_list, language_list


def __split_by_paragraph(text: str):
    """
    将长文本按段落划分。
    """
    text_list = text.split("\n")
    # 排除包含多个空格在内的空字符串
    text_list = [text.strip() for text in text_list if re.search(r"\S", text)]
    return text_list


def __split_by_sentence(text: str):
    """
    将单段落文本按句子划分。
    """
    text_list = re.split(r"[。\.\?\？\!\！]+", text)
    # 排除包含多个空格在内的空字符串
    text_list = [text.strip() for text in text_list if re.search(r"\S", text)]
    return text_list


def __split_by_within_sentence(text: str):
    """
    句子内停顿
    """
    text_list = re.split(r"[\;\；\、\,\，]+", text)
    text_list = [text.strip() for text in text_list if re.search(r"\S", text)]
    return text_list

def text_split_to_sentence(text: str) -> list:
    """
    将长文本按段落、句子、句内分别划分，生成三级列表
    """
    # 首先按段落划分
    text_paragraph_list = __split_by_paragraph(text=text)
    if len(text_paragraph_list) == 0:
        return []
    third_list = []
    for text_paragraph in text_paragraph_list:
        # 按句子划分
        secondary_list = []
        text_sentence_list = __split_by_sentence(text_paragraph)
        for text_sentence in text_sentence_list:
            # print(__split_by_within_sentence(text_sentence))
            secondary_list.append(__split_by_within_sentence(text_sentence))
        third_list.append(secondary_list)
    return third_list
