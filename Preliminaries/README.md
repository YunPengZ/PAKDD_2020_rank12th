#### 思路说明
本次大赛构建特征的思路主要有三个
1. 构建时间窗口内的统计特征（过去一天，三天，五天，一周，十五天内的'max','mean','std）
2. 构建数据变化率的特征（'diff'）
3. 使用原始数据

模型方面尝试了随机森林和lgbm,xgboost 其中rf在test a上表现最好 而lgbm在test b上表现最好

#### 代码运行说明 
../user_data/model_data/model.dat是已经训练好的模型 可以直接用于预测数据注：我使用joblib读可以使用)
对test数据集的处理，处理完的数据放再./user_data/tmp_data/test_data/test_v3.csv中，使用./code/main.py中use_feas的选择方式即可对数据进行预测

**/data文件夹中 按照代码规范里分为三种结构 round1_train中存放201707-2018007所有csv数据 round1_testB存放test b的数据 fault_tag直接放在./data下**
不知道我这样理解对不对 但是我模型中目前是按照这么处理的 

如果需要重新训练模型：需要先保证./data文件夹中包含赛题中的所有数据，并且结构一致，接着按照以下步骤处理

1.使用../feature/serial_process.ipynb处理得到serial_v2.csv 存放在../user_data/tmp_data/文件夹中 或者直接使用已处理好的文件

2.使用../feature/time_series_process.ipynb处理得到训练和数据 存放在../user_data/tmp_data/文件夹中 
两者均有对应的.py文件可以直接执行
3.运行 ../model/main.py文件 会将结果输出到../prediction_result/predictions.csv