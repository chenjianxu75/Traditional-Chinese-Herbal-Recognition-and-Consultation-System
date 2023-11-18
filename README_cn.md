## _传统中医药的识别_

传统中医作为世界医疗系统中不可或缺的医疗方法，目前越来越为人所知并被广泛使用。在此期间，传统中医也经历了飞速的发展。然而，中药材的种类繁多且复杂，普通人对于中药材的识别知识相对有限，这可能导致不当使用并带来不良后果。

- 由于不当使用而导致的中毒。
- 药效的丧失。
- 药物间的不良相互作用。
- 误导性的自我诊断，安全风险。
- ✨经济损失✨。

该模型的目的是帮助那些不了解或不熟悉传统中医的人，轻松识别传统中医药的种类，从而避免上述问题。

## 代码结构详解

> ├─ read.py # 数据读取
> ├─ split.py # 数据分类
> ├─ images  # 图像文件
> ├─ models # 模型
> ├─ main.py # 用户界面（此部分在上传时未完成）
> ├─ test.py # 模型测试
> ├─ train.py # 模型训练

## read.py
该代码的主要功能是分析指定文件夹下每个子文件夹中的图像数量，并使用Matplotlib库来创建横向柱状图，直观显示每个子文件夹中图像数量的分布。主要步骤包括：

- 使用 read_flower_data 函数计算每个子文件夹中的图像数量。
- 使用 show_bar 函数创建柱状图，Y轴显示子文件夹名称，X轴显示相应的图像数量。
- 该代码的主要目的是帮助用户理解特定文件夹中的图像数据，以更好地了解不同子文件夹中的图像数量分布。

以下是图像输出示例：
![示例图片](https://github.com/whossssssss/ML/blob/google-colab/myplot.png)

## split.py
该代码的主要功能是将源数据文件夹中的图像数据集分割成训练集、验证集和测试集，并将它们复制到不同的目标文件夹中。

data_set_split 函数：
- src_data_folder：源数据文件夹，包含每个类别的图像数据。
- target_data_folder：目标文件夹，用于存储分割后的数据集。
- train_scale, val_scale, test_scale：训练集、验证集和测试集的比例，默认为0.8、0.1、0.1。

函数内部操作：
- 首先，它提取源数据文件夹中所有类别（以文件夹形式存在）。
- 然后，在目标文件夹中创建三个子文件夹：train、val 和 test，分别用于存储训练集、验证集和测试集的图像数据。
- 接下来，它遍历每个类别：
  - 对于每个类别，它会随机打乱该类别的图像数据顺序。
  - 根据设定的比例将图像复制到训练集、验证集或测试集文件夹中。
  - 最后，函数输出每个类别的详细信息，包括文件夹路径和每个数据子集中的图像数量。

## train.py
此代码通过使用预训练的 MobileNetV2 模型作为基础模型进行迁移学习来选择模型。以下是模型的主要信息：
- 使用的预训练模型：MobileNetV2。
- 固定基础模型权重：`base_model.trainable = False`，意味着预训练模型 MobileNetV2 的权重被冻结，不进行微调。
- 自定义层：在 MobileNetV2 模型上添加的全局平均池化层和输出层。

以下是训练模型的图形输出：
![训练图像1](https://github.com/whossssssss/ML/blob/google-colab/train_1.png)
![训练图像2](https://github.com/whossssssss/ML/blob/google-colab/train_2.png)

在未来的模型改进计划中，我们会使用自定义的卷积神经网络（CNN）模型，但该模型在上传时尚未完成。

## test.py
加载训练好的模型并进行测试，以评估模型在测试数据上的表现。测试结果包括损失和准确率，这些指标衡量了模型对新的、以前未见过的数据的性能。

以下是测试集的输出数据：
```sh
Found 176 files belonging to 5 classes.
Using 35 files for validation.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 224, 224, 3)       0         
                                                                 
 mobilenetv2_1.00_224 (Func  (None, 7, 7, 1280)        2257984   
 tional)                                                         
                                                                 
 global_average_pooling2d (  (None, 1280)              0         
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 2264389 (8.64 MB)
Trainable params: 6405 (25.02 KB)
Non-trainable params: 2257984 (8.61 MB)
_________________________________________________________________
9/9 [==============================] - 2s 72ms/step - loss: 0.0044 - accuracy: 1.0000
Test accuracy : 1.0
```

用户界面 UI
这是初始的用户界面，系统会提示用户选择一张图片并进行预测。
![用户界面示例图片](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191311.png)
当模型接收到一张图片（支持常见的图片格式，如 jpg, jpeg, png 等），界面会对图像进行预测。
![用户界面示例图片](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191324.png)
![用户界面示例图片](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191331.png)
![用户界面示例图片](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191800.png)
![用户界面示例图片](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191807.png)
我们也将致力于探索更多功能，比如在输出时能够同时显示传统中药的治疗效果，以及训练和实施传统中医问答机器人（即它可以根据用户提供的症状进行相应的传统中医诊断等）。

后续内容将持续更新...


