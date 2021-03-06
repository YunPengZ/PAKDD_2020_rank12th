#### 比赛说明

【任务】需要对测试集中的硬盘按天粒度进行预测，判断该硬盘是否会在未来30天内发生故障

【难点分析】
1.故障样本占总样本的数据极少，因此样本不平衡问题极为严重
2.SMART数据原始特征纬度很多，有515维特征纬度，删除全空列后仍有53维特征
3.预测标签是未来30天内是否会发生故障，在定义正负样本时有些样本区别不大。

如：磁盘A在7月31号发生故障，那么7月1号磁盘A的数据为正样本。6月30号磁盘A的数据为负样本，但实际上两个样本间隔一天，理应区别不大。分别作为正负样本训练，模型会较难寻找分隔点。



#### 自己的思路说明
本次大赛构建特征的思路主要有两个
1. 构建数据变化率的特征,根据磁盘数据做shift 用变化量除以shift数据得到变化率（'diff_rate'），目的一是消除时间相关（与直接使用差分特征类似）二是去除量纲，并根据相关性筛选了一下变化率特征
2. 使用io类数据  241raw+242raw表示总共的I/O量
3. 对时间相关的特征做标准化 分别处以9raw/12raw 消除时间相关因素 做标准化操作
4. 数据采样方面：为了区分正负样本 我们对负样本做了部分选取，选取的规则是：

如果磁盘在未来的两个时间窗口（30天）内会发生故障 我们就不选择该磁盘作为负样本。

模型方面尝试了随机森林和lgbm,最终选定了rf

#### 部分反思 
初赛的思路是疯狂堆特征然后做特征筛选 模型融合这样 A榜一度到了第四 但是对数据的分析不够，导致shake的比较大 最终排到37th
复赛的时候在找实习 搞比赛的时间比较少 
但是转换了一下思路 重新对数据集做eda探索，也因此上了些分，只是可惜时间问题，很多东西还没来得及去做就结束了 不然还可以冲一下奖牌榜

这次比赛认知道的比较重要的事情是 对数据的认识、清洗、异常值处理、可视化分析往往比构造逻辑上可行的特征要重要的多。
还有就是尝试不同建模思路带来的差距比做特征要大的多啊...
模型融合真的就是听起来很美好 实际上 也只是在基本没思路的时候能带来一点微小的提升吧

还有就是华为竟然也有一个赛题一样的比赛...也是答辩的时候听别的选手说才知道,感觉错失了一大波分....
看了一下选手的思路 对时间数据的标准化操作有些相像。同时使用无监督模型处理得到了较好的分数，无监督模型进行磁盘异常检测也在华为实习生的口中也得到了验证，他们部门所做的正是无监督模型。
链接：[https://mp.weixin.qq.com/s/D0m6r7-jrNPulpv8UIgm7Q]

#### 代码运行说明 
../user_data/model_data/model.dat是已经训练好的模型 可以直接用于预测数据注：我使用joblib读可以使用)，直接调用./code/main.py文件即可 
但是测试集地址需要给定

###### test数据集的地址不太清楚该如何指定 因此直接按照提交的镜像地址给出 可能需要改一下 待改的位置在./code/readme.md中指出

如果需要重新训练模型
直接运行 ./code/main_rerun.py文件 会将结果输出到./code/文件夹下



##### 大佬的思路
一个显著区别就是使用回归来做，当时也考虑使用回归来做，但是对建模标签没有一个清晰的
还有一个比较重要的点是后处理操作：
对不同model分别选择磁盘 有点类似于多路召回 说明如果区别很大的不同类型 把类型作为直接输入到模型中得到的效果往往不如分别建模/分别输出；
