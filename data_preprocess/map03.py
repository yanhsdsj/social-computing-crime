import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------- 配置参数 ----------------------------
INPUT_FILE = 'data/ShanghaiCrimeProcessedFormat02.csv'  # 原始数据路径
OUTPUT_FILE = 'data/ShanghaiCrimeMapped03.csv'  # 处理后数据保存路径
MAPPING_FILE = 'data/CrimeTypeMapping.csv'
PLOT_THRESHOLD = 0.1  # 绘制阈值（仅显示比例≥0.1%的类型）
FONT_PATH = 'SimHei.ttf'  # 中文字体路径（系统需安装，如无则注释掉）

# ---------------------------- 犯罪类型归一化映射 ----------------------------
CRIME_TYPE_MAPPING = {
    # 主类型: [变体列表]（覆盖用户提供的115种类型中的主要变体）
    "盗窃": [
        "盗窃", "盗窃罪", "盗窃案", "盗窃一审刑事判决书",
        "盗窃罪,掩饰、隐瞒犯罪所得罪", "盗窃罪，故意伤害罪",
        "盗窃罪、偷越国（边）境罪", "盗窃罪、窝藏罪",
        "盗窃罪、信用卡诈骗罪", "盗窃罪、掩饰、隐瞒犯罪所得罪",
        "盗窃、信用卡诈骗", "盗窃、诈骗罪", "盗窃罪、脱逃罪",
        "盗窃罪、抢劫罪", "盗窃罪、敲诈勒索罪", "盗窃罪、诈骗罪",
        "盗窃罪，偷越国（边）境罪", "盗窃罪、危险驾驶罪", "盗窃罪和交通肇事罪"
    ],
    "抢劫": [
        "抢劫罪", "抢劫", "抢劫案", "抢劫罪、盗窃罪",
        "抢劫罪、贩卖毒品罪", "抢劫罪、抢夺罪", "抢劫罪、诈骗罪",
        "抢劫罪（最终认定为盗窃罪）", "抢劫罪、盗窃罪、交通肇事罪",
        "抢劫罪、寻衅滋事罪", "抢劫罪、强制猥亵罪", "抢劫罪、故意伤害罪",
        "抢劫罪、强奸罪", "抢劫罪，非法拘禁罪", "抢劫罪、非法拘禁罪",
        "抢劫罪、盗窃罪、掩饰、隐瞒犯罪所得罪", "抢劫罪、敲诈勒索罪"
    ],
    "诈骗": [
        "诈骗", "诈骗罪", "信用卡诈骗", "合同诈骗罪"
    ],
    "故意伤害": [
        "故意伤害", "故意伤害罪", "抢劫罪、故意伤害罪"  # 含跨类组合中的主行为
    ],
    "危险驾驶": [
        "危险驾驶", "危险驾驶罪"
    ],
    "寻衅滋事": [
        "寻衅滋事", "寻衅滋事罪", "寻衅滋事一审刑事判决书", "寻衅滋事案",
        "寻衅滋事罪、盗窃罪"
    ],
    "交通肇事": ["交通肇事罪"],
    "贩卖毒品": ["贩卖毒品", "贩卖毒品罪", "运输毒品罪", "非法持有毒品罪"],
    "强奸": ["强奸罪", "强奸罪、抢劫罪", "抢劫罪、强奸罪"],
    "其他": [  # 无法归入以上主类的剩余类型（可根据需求扩展主类）
        "刑事", "刑事判决", "刑事判决书", "民事纠纷", "民事",
        "行政诉讼", "行政", "行政判决", "行政强制", "航空旅客运输合同纠纷",
        "财产保险合同纠纷", "机动车交通事故责任纠纷", "财产损害赔偿纠纷",
        "监护人责任纠纷", "身体权纠纷", "生命权纠纷", "物业服务合同纠纷",
        "生命权、健康权、身体权纠纷", "劳动合同纠纷", "财产损失保险合同纠纷",
        "抵押权纠纷", "名誉权纠纷", "房屋买卖合同纠纷", "运输合同纠纷",
        "旅游合同纠纷", "买卖合同纠纷", "借记卡纠纷", "房屋租赁合同纠纷",
        "电信服务合同纠纷", "责任保险合同纠纷", "公路货物运输合同纠纷",
        "民间借贷纠纷", "身体权、健康权纠纷", "保险人代位求偿权纠纷",
        "保管合同纠纷", "申请撤销仲裁裁决", "违反安全保障义务责任纠纷"
    ]
}

ENGLISH_COLUMNS = {
    "案号": "case_number",
    "犯罪类型": "crime_type",
    "犯罪类型ID": "crime_type_id",
    "区名": "district_name",
    "区ID": "district_id",
    "纬度": "latitude",
    "经度": "longitude",
    "案发时间": "incident_time",
    "判决时间": "judgment_date"
}

# ---------------------------- 映射字典 ----------------------------
shanghai_districts = {
    "黄浦区": 1, "徐汇区": 2, "长宁区": 3, "静安区": 4, "普陀区": 5, "虹口区": 6, "杨浦区": 7,
    "闵行区": 8, "宝山区": 9, "嘉定区": 10, "浦东新区": 11, "金山区": 12, "松江区": 13,
    "青浦区": 14, "奉贤区": 15, "崇明区": 0
}

court_to_district = {
    '上海市浦东新区人民法院': '浦东新区', '上海市闵行区人民法院': '闵行区', '上海市松江区人民法院': '松江区',
    '上海市嘉定区人民法院': '嘉定区', '上海市宝山区人民法院': '宝山区', '上海市奉贤区人民法院': '奉贤区',
    '上海市静安区人民法院': '静安区', '上海市黄浦区人民法院': '黄浦区', '上海市青浦区人民法院': '青浦区',
    '上海市徐汇区人民法院': '徐汇区', '上海市杨浦区人民法院': '杨浦区', '上海市金山区人民法院': '金山区',
    '上海市普陀区人民法院': '普陀区', '上海市长宁区人民法院': '长宁区', '上海铁路运输法院': '静安区',
    '上海市虹口区人民法院': '虹口区', '上海市崇明区人民法院': '崇明区', '崇明县人民法院': '崇明区',
    '上海市第一中级人民法院': '长宁区', '上海市第二中级人民法院': '静安区', '上海市第三中级人民法院': '浦东新区',
    '上海铁路运输中级法院': '虹口区', '上海市闸北区人民法院': '静安区', '上海市崇明县人民法院': '崇明区',
    '上海海事法院': '浦东新区'
}


# ---------------------------- 核心处理函数 ----------------------------
def normalize_crime_type(case_type: str) -> str:
    """将犯罪类型变体归一化为主类型（如"盗窃罪"→"盗窃"）"""
    for main_type, variants in CRIME_TYPE_MAPPING.items():
        if case_type.strip() in variants:
            return main_type
    return "其他"


def crime_id_mapping(df: pd.DataFrame, filename: str):
    """保存犯罪类型与ID的映射关系"""
    unique_types = df[['crime_type', 'crime_type_id']].drop_duplicates()
    mapping = dict(zip(unique_types['crime_type'], unique_types['crime_type_id']))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("犯罪类型\t类型ID\n")
        for crime_type, type_id in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{crime_type}\t{type_id}\n")


def process_crime_data(input_file: str, output_file: str) -> pd.DataFrame:
    # 1. 读取原始数据
    df = pd.read_csv(input_file)

    # 2. 犯罪类型归一化
    df['crime_type'] = df['case_type'].apply(normalize_crime_type)

    # 3. 利用法院名称映射到 district_id
    df['district_name'] = df['court_name'].map(court_to_district)
    df['district_id'] = df['district_name'].map(shanghai_districts)

    # 4. 删除多余字段
    df = df.drop(columns=['case_type', 'court_name'])

    # 5. 筛选：保留除“其他”和“交通肇事”外的犯罪类型
    df = df[~df['crime_type'].isin(["其他", "交通肇事"])]

    # 6. 生成犯罪类型 ID
    type_id_map = {ctype: idx for idx, ctype in enumerate(sorted(df['crime_type'].unique()))}
    df['crime_type_id'] = df['crime_type'].map(type_id_map)

    # 7. 保存数据和映射关系
    df.to_csv(output_file, index=False)
    crime_id_mapping(df, MAPPING_FILE)

    return df



def visualize_crime_distribution(df: pd.DataFrame):
    """根据归一化后的 crime_type 绘制犯罪类型分布图"""
    # 统计归一化后犯罪类型分布（百分比）
    crime_dist = df['crime_type'].value_counts(normalize=True) * 100

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = ['SimHei']
    bars = plt.bar(crime_dist.index, crime_dist.values)

    plt.title('上海市犯罪类型分布（归一化后，百分比）')
    plt.xlabel('犯罪类型')
    plt.ylabel('百分比（%）')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('imgs/crime_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_crime_types(df: pd.DataFrame, threshold: float) -> None:
    """对归一化后的犯罪类型进行阈值过滤并绘制图表"""
    # 计算分布
    distribution = df['crime_type'].value_counts(normalize=True) * 100

    # 过滤低比例类型
    filtered = distribution[distribution >= threshold].copy()
    others = distribution[distribution < threshold].sum()
    if others > 0:
        filtered['其他'] = others

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = ['SimHei']
    bars = plt.bar(filtered.index, filtered.values)

    plt.title('上海市犯罪类型分布（阈值合并后）')
    plt.xlabel('犯罪类型')
    plt.ylabel('占比（%）')
    plt.xticks(rotation=30, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('imgs/crime_distribution_threshold.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 处理数据并可视化
    df_processed = process_crime_data(INPUT_FILE, OUTPUT_FILE)
    visualize_crime_distribution(df_processed)
    analyze_crime_types(df_processed, PLOT_THRESHOLD)
