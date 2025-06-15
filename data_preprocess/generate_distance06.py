import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# 创建地理编码器对象
geolocator = Nominatim(user_agent="shanghai_neighborhoods")

# 上海的区及对应的映射数字
shanghai_districts = {
    "黄浦区": 1,
    "徐汇区": 2,
    "长宁区": 3,
    "静安区": 4,
    "普陀区": 5,
    "虹口区": 6,
    "杨浦区": 7,
    "闵行区": 8,
    "宝山区": 9,
    "嘉定区": 10,
    "浦东新区": 11,
    "金山区": 12,
    "松江区": 13,
    "青浦区": 14,
    "奉贤区": 15,
    "崇明区": 0
}

# 获取每个区的中心坐标
district_coordinates = {}
for district, number in shanghai_districts.items():
    location = geolocator.geocode(f"{district}, 上海市")
    if location:
        district_coordinates[number] = (location.latitude, location.longitude)
    else:
        print(f"未能获取 {district} 的坐标")

# 创建数据框存储距离
districts = list(shanghai_districts.values())
num_districts = len(districts)
df = pd.DataFrame(index=districts, columns=districts)

# 计算各区域之间的距离
for i in range(num_districts):
    for j in range(i, num_districts):
        if i in district_coordinates and j in district_coordinates:
            coord1 = district_coordinates[i]
            coord2 = district_coordinates[j]
            distance = geodesic(coord1, coord2).kilometers  # 计算两点之间的距离（公里）
            df.iloc[i, j] = distance
            df.iloc[j, i] = distance
        else:
            print(f"未能计算 {i} 和 {j} 之间的距离，可能缺少坐标数据")

# 将距离数据转换为列表形式
data = []
for i in range(num_districts):
    for j in range(num_districts):
        if i in district_coordinates and j in district_coordinates:
            data.append([districts[i], districts[j], df.iloc[i, j]])
        else:
            data.append([districts[i], districts[j], np.nan])  # 缺少坐标数据时用 NaN 表示

# 创建完整的数据框
columns = ['neighborhood', 'neighborhood-2', 'st_distance']
df_final = pd.DataFrame(data, columns=columns)

# 标准化处理距离数据（Min-Max 缩放）
df_final_nonan = df_final.dropna(subset=['st_distance'])  # 去除缺少距离数据的行
min_distance = df_final_nonan['st_distance'].min()
max_distance = df_final_nonan['st_distance'].max()
df_final_nonan['st_distance_standardized'] = (df_final_nonan['st_distance'] - min_distance) / (max_distance - min_distance)

# 将结果导出为 CSV 文件
df_final_nonan.to_csv('data/shanghai_neighborhood_distances_standardized.csv', index=False)

print("CSV 文件已生成！")