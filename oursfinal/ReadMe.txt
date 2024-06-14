DBT.py为模型代码
display.py为调参运行代码
其余为数据读取、展示、分析的相关代码

首先在display.py选择想要读取的数据，并设定好batch_size等相关参数
之后在终端中指定目录下运行python display.py即可运行代码

datasets文件中缺失的magnet数据集可从SGT模型研发者的GITHUB中下载

packages：numpy pandas scipy tqdm matplotlib scikit-learn torch
