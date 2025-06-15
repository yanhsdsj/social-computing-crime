# CrimeForecaster+ 项目

## 一、实验任务

本项目旨在基于时空数据，利用深度学习模型预测城市犯罪类型的发生。我们将结合图卷积网络（GCN）与门控循环单元（GRU），通过时空模型有效捕捉犯罪事件的空间与时间依赖关系，进一步提升犯罪预测精度。在此基础上，结合 **TimesNet** 模型的思想，优化了模型的时序特征提取能力。通过对上海市真实犯罪数据集的实验验证，展示了改进模型在实际城市环境中的应用效果。

## 二、依赖

实验环境依赖已列在 `requirements.txt` 文件中。您可以执行以下命令安装所有依赖：

```bash
pip install -r requirements.txt
```

## 三、文件夹结构

```bash
+-- data                          数据集，包含多个城市的训练集、验证集和测试集
+-- data_preprocess               数据预处理脚本，包含中间数据
+-- lib                            工具库，包含辅助函数和类
+-- model                          设计的模型文件夹，包含我们根据改进思路设计的模型
|   +-- graph_conv.py              图卷积层的实现
|   +-- graph_times_net.py         结合图卷积与时间序列网络的实现
|   +-- inception_block.py         Inception模块的实现
|   +-- times_block.py             TimesNet相关模块的实现
+-- origin                         原始模型文件夹，包含原始模型的全部实现
+-- result                         只使用timesnet模型的预测结果
+-- opti_inception_with_v2_result  优化后的模型输出，包含最终的优化结果
+-- run.py                         训练与评估的主要运行脚本
+-- README.md                      项目说明文档
+-- requirements.txt               环境依赖
+-- supervisor.py                  训练与评估核心代码
```

## 四、运行代码

### 4.1 数据预处理

在开始训练模型之前，需要进行数据的预处理。预处理过程将在 `data_preprocess` 文件夹中完成。此过程包括数据清洗、格式化，以及生成用于训练、验证和测试的数据集。

### 4.2 训练模型

数据预处理完成后，可以开始训练模型。训练脚本位于 `run.py` 文件中。该脚本将使用处理后的数据，按设置的超参数进行模型训练，并保存在指定的文件夹中。

### 4.3 模型评估

训练完成后，模型将进入评估阶段。评估过程在 `supervisor.py` 文件中定义，主要用于计算模型在测试集上的各项指标（如 **F1-score**、**Precision**、**Recall** 等）。评估结果将输出至 `result` 文件夹中，供进一步分析与验证。

## 五、参考

本项目参考了以下文献资料：

- Sun, J., Yue, M., Lin, Z., Yang, X., Nocera, L., Kahn, G., & Shahabi, C. (2020). CrimeForecaster: Crime prediction by exploiting the geographical neighborhoods' spatiotemporal dependencies. *ECML/PKDD*.
- Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. In *Proceedings of the International Conference on Learning Representations (ICLR)*.
- [A dataset on the spatiotemporal distributions of street and neighborhood crime in China](https://figshare.com/articles/dataset/_b_A_dataset_on_the_spatiotemporal_distributions_of_street_and_neighborhood_crime_in_China_b_/28106939)
- [Biotrade news on crime analysis](https://www.ebiotrade.com/newsf/2025-3/20250321061506298.html)

本项目采用以下评估指标来衡量模型性能：

- **Macro F1**：关注模型对低频犯罪类型的预测能力。
- **Micro F1**：衡量模型整体预测准确性。
- **Precision**：计算每个类别的精确度。
- **Recall**：计算每个类别的召回率。

------

## 六、结论

通过将 **CrimeForecaster** 与 **TimesNet** 模型相结合，我们在犯罪预测中取得了显著的性能提升，特别是在稀有犯罪类别的预测上。实验结果表明，基于这种改进方法的模型不仅能提高预测准确率，还能有效捕捉时空特征的依赖关系，展示出良好的适用性与优化潜力。这种方法能够在城市治理和治安预警系统中发挥重要作用。
