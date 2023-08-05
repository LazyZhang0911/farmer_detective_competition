# 1.导入所需模块

```python
import glob
import time
import cv2
import numpy as np
import pandas as pd
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.dataset import Dataset
```
# 2.查看本地 cuda 是否可用，并引入比赛数据集

```python
# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 读取数据集
train_path = glob.glob('alldata\\train\\*')
test_path = glob.glob('alldata\\test\\*')
train_path.sort()
test_path.sort()
train_df = pd.read_csv('alldata\\train.csv')
train_df = train_df.sort_values(by='name')
train_label = train_df['label'].values
# 自定义数据集
# 带有图片缓存的逻辑
DATA_CACHE = {}

class XunFeiDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = cv2.imread(self.img_path[index])
            DATA_CACHE[self.img_path[index]] = img
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(self.img_label[index]))
    def __len__(self):
        return len(self.img_path)
```
# 3.设置训练、验证、测试数据集的各个属性

```python
import albumentations as A
# 训练集
train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-1000], train_label[:-1000],
                  A.Compose([
                      A.RandomRotate90(),
                      A.Resize(256, 256),
                      A.RandomCrop(224, 224),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                  ])
                  ), batch_size=32, shuffle=True, num_workers=0, pin_memory=False
)
# 验证集
val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-1000:], train_label[-1000:],
                  A.Compose([
                      A.Resize(256, 256),
                      A.RandomCrop(224, 224),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                  ])
                  ), batch_size=30, shuffle=False, num_workers=0, pin_memory=False
)
# 测试集
test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path, [0] * len(test_path),
                  A.Compose([
                      A.Resize(256, 256),
                      A.RandomCrop(224, 224),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                  ])
                  ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)
```

# 4.设置神经网络结构设置及搭建

```python
class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
        model = models.resnet50(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        # model.fc1 = nn.Linear(16*2048, 1000)
        # model.fc2 = nn.Linear(1000, 512)
        model.fc3 = nn.Linear(512, 25)
        self.resnet = model
    def forward(self, img):
        out = self.resnet(img)
        return out

model = XunFeiNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), 0.001)
print(model)
```
# 5.设置模型训练、验证、测试函数

```python
# 模型训练
def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input).to(device)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train loss', loss.item())
        train_loss += loss.item()
    return train_loss / len(train_loader)

# 模型验证
def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            output = model(input).to(device)
            loss = criterion(output, target)
            val_acc += (output.argmax(1) == target).sum().item()
    return val_acc / len(val_loader.dataset)

# 模型预测
def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0
    test_pred = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            output = model(input).to(device)
            test_pred.append(output.data.cpu().numpy())
    return np.vstack(test_pred)
```
# 6.设置训练次数，进行模型训练并验证，生成提交文件

```python
if __name__ == '__main__':
    print('strat training...')
    for i in range(31):
        print(f'epoch{i} is training...')
        train_loss = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)
        train_acc = validate(train_loader, model, criterion)
        if i % 10 == 0:
            print('train finish', 'train_loss:', train_loss, 'train_acc:', train_acc, 'test_acc:', val_acc)
    # 对测试集多次预测
    pred = None
    for _ in range(3):
        if pred is None:
            pred = predict(test_loader, model, criterion)
        else:
            pred += predict(test_loader, model, criterion)
    submit = pd.DataFrame(
        {
            'name': [x.split('/')[-1] for x in test_path],
            'label': pred.argmax(1)
        })

    # 生成提交结果
    submit = submit.sort_values(by='name')
    submit.to_csv('submit.csv', index=None)

for _ in range(3):
    if pred is None:
        pred = predict(test_loader, model, criterion)
    else:
        pred += predict(test_loader, model, criterion)
submit = pd.DataFrame(
    {
        'name': [x.split('/')[-1] for x in test_path],
        'label': pred.argmax(1)
    })
# 生成提交结果
submit = submit.sort_values(by='name')
submit.to_csv('submit.csv', index=None)
```

