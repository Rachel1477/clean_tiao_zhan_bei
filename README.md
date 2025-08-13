# 操作说明文档
本文档包含文件目录结构，文件功能的概括，以及如何使用本项目

# 项目目录如下
```
tree
.
|── 点迹                        <--------存放点迹数据
|── 航迹                        <--------存放航迹数据
|── 原始回波                     <--------存放原始回波
|── 测试数据                     <--------这是为比赛测试专门预留的通道！！
├── gradio_src                  <--------这是跟gradio可视化相关的代码文件夹
│   ├── gradio_interface.py
│   ├── run_prediction_process.py
│   └── run_training_process.py
└── templates                   
|    ├── index.html           
├── src                         <--------核心代码所在的文件夹
│   ├── model_save
│   │   ├── best.pth
│   │   ├── label_map.json
│   │   └── tabular_scaler.npz
│   ├── online_prediction_results
│   │   ├── online_confusion_matrix.png
│   │   └── online_prediction_details.csv
│   ├── apply.py
│   ├── model.py
│   ├── preprocess.py
│   ├── scoring.py
│   ├── train.py
│   ├── utils.py
│   └── valid.py
│   └── get_DR_fortest.m
│   └── get_DR_fortrain.m
├── myenv.yml
├── app.py
├── split_dataset.py

```

## 2.对于文件的解释
### src(核心代码)
- preprocess.py  ：定义了所有的预处理操作，包括DR map的加载，标准化处理，处理时间变量，构建多模态数据组合等
- model.py : 定义了模型的结构
- train.py ：训练脚本，直接运行即可开始训练（启动前请确定已正确划分训练集和验证集）
- valid.py ：验证脚本，用于评估模型能力，会使用测试集数据进行验证（启动前请确定已正确划分测试集）
- scoring.py : 对valid.py的输出结果进行打分。脚本会按照赛题组的评估标准计算准确率，有效周期，并输出出错批号，方便发现问题
- apply.py ：使用训练练好的模型应用到未知数据。
- get_DR_fortrain.m ：用于从原始回波中得到DR map数据，并保存为npy格式。会保留label
- get_DR_fortest.m  ：用于从原始回波中得到DR map数据，并保存为npy格式。不会保留label

### 2.gradio_src（gradio可视化）
>为了方便操作，本项目提供一个可视化操作窗口，用gradio实现
- gradio_interface.py ：总接口，运行它可以启动可视化操作窗口（包括训练和应用于待测试数据）
- run_prediction_process.py ：定义了应用于待测试数据相关的操作
- run_training_process.py ：定义了与模型训练相关的操作

### 3.其他
- app.py :用flask搭建的后端，对应templates下的index.html，有一个可以展示数据的界面，包括雷达图，点迹分布等
- split_dataset.py ：用于自动划分数据集，共划分为3类，分别是train,val,test,默认比例为7：1.5：1.5，可自定义                  

## 3.快速开始
### (1).准备数据集
你的数据集必须包含下面三个子文件夹
```
|── 点迹                   
|── 航迹                       
|── 原始回波  
```
然后打开matlab运行**get_DR_fortrain.m**，它会从.dat中提取出DR map并用npy格式保存

随后
```
python split_dataset.py
```

它会在本地生成一个DRmap文件夹，并按比例划分数据
至此，数据集准备完毕

### (2).依赖安装
本项目涉及到matlab和python两种语言。

1.安装matlab

从这里[下载](https://www.mathworks.com/products/matlab.html)并安装


2.安装python依赖
本项目提供两种方式
你可以
```
pip install -r requirement.txt
```
或者下载conda-pack打包好的环境,下载并解压到你的电脑，然后运行
```
conda-unpack
activate
```

### (3).训练
```
python train.py
```
模型保存在同级目录的model_save目录下
### (4).模型评估
```
python valid.py
python scoring.py
```
预测结果会保存在同级目录的online_prediction_results目录下
### (5).应用到自定义数据
你需要像(1).准备数据集那样准备数据集。

同样的你的数据集必须包含下面三个子文件夹
```
|── 点迹                   
|── 航迹                       
|── 原始回波  
```
随后打开matlab运行**get_DR_fortrain.m**，它会从.dat中提取出无label格式DR map并用npy格式保存

接着运行
```
python apply.py
```
预测结果会直接写回你的数据集中的航迹文件，并添加在最后一列。同时同级目录test_results_output下也会生成备份方便后续工作

## 4.可视化操作界面
你也可以直接在gradio界面上进行操作
```
python gradio_interface.py
```
随后浏览器打开http://127.0.0.1:7860 即可访问
![pic1](pic/pic1.png)
(训练界面，各种参数均可调节，点击开始训练按钮即可正常训练)
！[pic2](pic2)
(填入正确路径即可开始预测)

>受限与gradio不支持系统原生的文件目录管理器，这里只能手动填入路径。

## 5.数据展示
```
python app.py
```
随后浏览器打开 http://127.0.0.1:5000 即可访问
！[pic3](pic/pic3.png)

该界面支持所有批号选择（包括训练数据和待测试数据），会展示雷达航迹图，对应点迹分布，批号的具体数据，预测结果（待测试数据）。支持图片比例尺调节，路径选择，放大查看（在雷达图上长按左键）

