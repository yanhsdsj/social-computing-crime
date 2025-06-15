import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Optional, Tuple
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# 配置参数
TIME_FORMATS = [
    '%Y-%m-%d %H:%M:%S',
    '%Y/%m/%d %H:%M',
    '%Y年%m月%d日%H时%M分',
    '%Y年%m月%d日%H:%M',
    '%Y.%m.%d %H:%M'
]

CHINESE_NUM_MAP = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16,
    '十七': 17, '十八': 18, '十九': 19, '二十': 20, '二十一': 21, '二十二': 22, '二十三': 23,
    '二十四': 24, '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29,
    '三十': 30, '三十一': 31, '元': 1, '正': 1, '腊': 12, '冬': 11
}

XUN_MAP = {
    '上旬': (1, 10), '初旬': (1, 10),
    '中旬': (11, 20),
    '下旬': (21, 31), '末旬': (21, 31)
}

SEASON_MAP = {
    '上半年': (1, 1, 6, 30),
    '下半年': (7, 1, 12, 31),
    '一季度': (1, 1, 3, 31),
    '二季度': (4, 1, 6, 30),
    '三季度': (7, 1, 9, 30),
    '四季度': (10, 1, 12, 31)
}


def parse_time_component(s: str, default: int = 0, max_val: int = None) -> int:
    """解析时间成分并验证有效性"""
    try:
        num = int(s)
        if max_val is not None:
            num = min(max(num, 0), max_val)
        return num
    except:
        return default


def parse_datetime_precise(time_str: str) -> Optional[datetime]:
    """精确解析包含时分的时间字符串"""
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    return None


def parse_chinese_date_part(s: str, is_month: bool = False) -> Optional[int]:
    """解析中文日期部分（月/日），返回数字"""
    if not s:
        return None

    # 直接数字解析
    if s.isdigit():
        num = int(s)
        if is_month and 1 <= num <= 12:
            return num
        elif not is_month and 1 <= num <= 31:
            return num
        return None

    # 中文解析
    if s in CHINESE_NUM_MAP:
        num = CHINESE_NUM_MAP[s]
        if is_month and 1 <= num <= 12:
            return num
        elif not is_month and 1 <= num <= 31:
            return num

    # 处理复合中文数字（如"二十一"）
    if len(s) > 1:
        try:
            parts = re.findall(r'[上下]|\d+|[^上下\d]', s)
            if '十' in parts:
                base = 10
                if len(parts) == 1:  # 十
                    return 10
                elif parts[0] in ('二', '两'):  # 二十
                    base = 20
                elif parts[0] == '三':  # 三十
                    base = 30
                add = sum(CHINESE_NUM_MAP.get(p, 0) for p in parts[1:])
                return base + add
        except:
            pass
    return None


def generate_random_date(start: datetime, end: datetime) -> datetime:
    """生成指定时间范围内的随机日期"""
    time_between = end - start
    random_seconds = random.randint(0, int(time_between.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def parse_date_range(text: str) -> Optional[Tuple[datetime, datetime]]:
    """解析复杂中文日期范围描述"""
    text = re.sub(r'[\s前后期间左右约]+', '', text)

    # 匹配旬范围模式（2016年7月中旬至8月下旬）
    xun_pattern = r'''
        (\d{4})年                                 # 开始年份
        ([\u4e00-\u9fa5\d]{1,3})月?               # 开始月份
        ([上中下]旬)至                            # 开始旬
        (\d{4})年                                 # 结束年份
        ([\u4e00-\u9fa5\d]{1,3})月?               # 结束月份
        ([上中下]旬)                              # 结束旬
    '''
    match = re.search(xun_pattern, text, re.VERBOSE)
    if match:
        try:
            # 修正括号匹配问题
            start_year = int(match.group(1))  # 移除多余括号
            start_month = parse_chinese_date_part(match.group(2), is_month=True)  # 修正参数传递
            start_xun = XUN_MAP.get(match.group(3), (1, 10))  # 修正括号匹配

            end_year = int(match.group(4))
            end_month = parse_chinese_date_part(match.group(5), is_month=True)
            end_xun = XUN_MAP.get(match.group(6), (21, 31))  # 修正括号匹配

            # 添加变量有效性校验
            if None in (start_month, end_month):
                return None

            start_date = datetime(start_year, start_month, start_xun[0])
            end_date = datetime(end_year, end_month, end_xun[1])
            return start_date, end_date
        except (ValueError, TypeError) as e:
            print(f"日期范围解析错误: {str(e)}")
            return None


def parse_datetime_extended(original: str) -> Optional[datetime]:
    """增强版日期时间解析"""
    if pd.isna(original) or not original.strip():
        return None

    # 先尝试精确解析
    dt = parse_datetime_precise(original.strip())
    if dt:
        return dt

    # 处理日期范围
    date_range = parse_date_range(original)
    if date_range:
        return generate_random_date(*date_range)

    # 修复后的正则表达式（完整闭合所有括号）
    time_pattern = r'''
        (\d{4})年                                # 年份
        (?:(\d{1,2}|[\u4e00-\u9fa5]+)月?)        # 月份（非捕获组）
        (?:(\d{1,2}|[\u4e00-\u9fa5]+)[日号]?)    # 日期（非捕获组）
        (?:\u65f6?(\d{1,2})[:点](\d{1,2})分?)?   # 时间（非捕获组）
    '''

    match = re.search(time_pattern, original, re.VERBOSE)
    if match:
        try:
            year = int(match.group(1))
            month = parse_chinese_date_part(match.group(2), is_month=True) or random.randint(1, 12)
            day = parse_chinese_date_part(match.group(3)) or random.randint(1, 28)
            hour = parse_time_component(match.group(4), default=random.randint(0, 23), max_val=23)
            minute = parse_time_component(match.group(5), default=random.randint(0, 59), max_val=59)

            # 日期验证
            day = min(day, 31)
            if month == 2:
                day = min(day, 28)
            return datetime(year, month, day, hour, minute)
        except:
            pass

    # 最终尝试pandas解析
    try:
        return pd.to_datetime(original, errors='raise', format='mixed')
    except:
        return None


def process_crime_data(input_file: str, output_file: str):
    """主处理函数"""
    df = pd.read_csv(input_file)

    # 转换日期列
    df['judgment_date'] = df['judgment_date'].apply(parse_datetime_extended)
    df['incident_time'] = df['incident_time'].apply(parse_datetime_extended)

    # 严格空值处理：只有当incident_time为null时才用judgment_date填充
    mask = df['incident_time'].isna()
    df.loc[mask, 'incident_time'] = df.loc[mask, 'judgment_date']

    # 删除仍然为空的记录
    df = df.dropna(subset=['incident_time'])

    # 类型一致性检查
    if not is_datetime(df['incident_time']):
        df['incident_time'] = pd.to_datetime(df['incident_time'])

    if not is_datetime(df['judgment_date']):
        df['judgment_date'] = pd.to_datetime(df['judgment_date'])

    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"处理完成，有效记录数：{len(df)}，保存路径：{output_file}")


if __name__ == "__main__":
    process_crime_data(
        input_file='data/ShanghaiCrimeData01.csv',
        output_file='data/ShanghaiCrimeProcessedFormat02.csv'
    )