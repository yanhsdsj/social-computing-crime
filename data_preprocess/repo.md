### 数据处理过程：
1. ChinaCrimeData -> ShanghaiCrimeData01(select_city_data01.py 通过字段"case_number"提取上海数据)
2. ShanghaiCrimeData01 -> ShanghaiCrimeProcessedFormat02(time_preprocess02.py 处理字段""crime_time"标准化时间格式)
3. ShanghaiCrimeData02 -> ShanghaiCrimeMapped03(map03.py 处理字段""court_name"得到区级地理位置及其数字映射、""crime_type"也有对应数字映射)
4. ShanghaiCrimeData03 -> **ShanghaiCrimeData04**(gen_crime_pkl04.py 将数据和LA数据进行对齐并生成.pkl文件)
5. ShanghaiCrimeData05 -> CRIME-SHANGHAI/ (gen_dataset05.py 生成时间窗口等数据，生成最终的训练集、验证集、测试集)
5. shanghai_district_distances06 (generate_distance06.py 得到各区距离，如果已有文件，不必重复生成)
5. shanghai_district_distances06 -> **adj_mx_shanghai07.pkl** (my_adj_mx07.py 得到邻接矩阵)
6. 