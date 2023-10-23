import torchvision
from torch import nn
import os
import json
import pickle

import torch
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

epochs = 10
lr = 0.03
batch_size = 32
image_path = 'C:/Users/17811/Desktop/task/tar'
model_path = './checkpoints/resnet101-5d3b4d8f.pth'
save_path = 'C:/Users/17811/Desktop/task/model'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1.数据转换
data_transform = {
    # 训练中的数据增强和归一化
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值方差归一化
    ]),
    # 验证集不增强，仅进行归一化
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2.形成训练集
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                     transform=data_transform['train'])

# 3.形成迭代器
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

print('using {} images for training.'.format(len(train_dataset)))

# 4.建立分类标签与索引的关系
cloth_list = train_dataset.class_to_idx
class_dict = {v: k for k, v in cloth_list.items()}
with open('class_dict.pk', 'wb') as f:
    pickle.dump(class_dict, f)

# 5.加载shufflenet模型
model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)  # 加载预训练好的shufflenet模型

# 将模型加载到GPU
model = model.to(device)

# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层的全连接层
model.fc = nn.Linear(model.fc.in_features, 5)

criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

# 6.模型训练
best_acc = 0  # 最优精确率
best_model = None  # 最优模型参数

for epoch in range(epochs):
    model.train()
    running_loss = 0  # 损失
    correct_count = 0  # 正确的样本数
    total_count = 0  # 总的样本数
    train_bar = tqdm(train_loader)

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_count += labels.size(0)
        correct_count += (predicted == labels).sum().item()
        train_bar.set_description(f"train epoch[{epoch + 1}/{epochs}] loss: {loss.item():.3f}")

    # 计算训练精度
    train_accuracy = 100 * correct_count / total_count

    # 打印信息
    print(f"【EPOCH: 】 {epoch + 1}")
    print(f"training loss： {running_loss}")
    print(f"training accuracy：  {train_accuracy:.2f}%")

    # 保存最优模型
    if train_accuracy > best_acc:
        best_acc = train_accuracy
        best_model = model.state_dict()

# 保存模型
torch.save(best_model, save_path)

print('Finished Training')

# 加载索引与标签映射字典
with open('class_dict.pk', 'rb') as f:
    class_dict = pickle.load(f)

# 数据变换
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 图片路径
img_path = r'./data/train/dangshen/dangshen_206.jpg'

# 打开图像
img = Image.open(img_path)

# 对图像进行变换
img = data_transform(img)

plt.imshow(img.permute(1, 2, 0))
plt.show()

# 将图像升维，增加batch_size维度
img = torch.unsqueeze(img, dim=0)

# 获取预测结果
output = model(img.to(device))
prediction = class_dict[output.argmax(1).item()]
