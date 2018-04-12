
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np

from skimage import io,transform
import matplotlib.pyplot as plt

import torch.nn
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

ROOT = "Datasets/corel_5k/images/"
dirs = [ROOT+i+"/" for i in next(os.walk(ROOT))[1]]
files = []
[files.extend([i+j for j in next(os.walk(i))[2] if "jpeg" in j]) for i in dirs]

with open("Datasets/corel_5k/labels/training_label") as f:
    train_labels = f.readlines()
train_labels = [i.split(" ")[:] for i in train_labels]
train_labels = [[int(j) for j in i if j != '' and j != '\n']for i in train_labels]
random.shuffle(train_labels)
train_label = train_labels[:4000]
val_label = train_labels[4000:]

train_label_dict = {}
for i in train_label:
    train_label_dict[str(i[0])+".jpeg"] = i[1:]
    
val_label_dict = {}
for i in val_label:
    val_label_dict[str(i[0])+".jpeg"] = i[1:]
    
with open("Datasets/corel_5k/labels/test_label") as f:
    test_labels = f.readlines()
test_labels = [i.split(" ")[:] for i in test_labels]
test_labels = [[int(j) for j in i if j != '' and j != '\n']for i in test_labels]
test_label_dict = {}
for i in test_labels:
    test_label_dict[str(i[0])+".jpeg"] = i[1:]


# In[2]:


train_pairs = []
val_pairs = []
test_pairs = []
for i in files:
    img_name = i.split("/")[-1]
    if img_name in val_label_dict.keys():
        val_pairs.append((i, val_label_dict[img_name]))
    elif img_name in test_label_dict.keys():
        test_pairs.append((i, test_label_dict[img_name]))
    elif img_name in train_label_dict.keys():
        train_pairs.append((i, train_label_dict[img_name]))


# In[3]:


class COREL_5K(Dataset):
    def __init__(self, data, num):
        super(COREL_5K, self).__init__()
        self.data = data
        self.num = num
        self.mean = [24.534070819674163, 25.25100188727893, 21.32722067148645]
    
    def __getitem__(self, index):
        data_path, label = self.data[index]
        label = np.array(label) - 1
        img = io.imread(data_path)
        if img.shape != (192, 128, 3):
            img = transform.resize(img, (192, 192))
            img = transform.rotate(img, 90)
            img = transform.resize(img, (192, 128))
        img = np.array(img)
        img[:, :, 0] = img[:, :, 0] - self.mean[0]
        img[:, :, 1] = img[:, :, 1] - self.mean[1]
        img[:, :, 2] = img[:, :, 2] - self.mean[2]
        img = img.astype(np.float32)
        
        label = np.sum(np.eye(374)[label], axis=0)
        return img, label.astype(np.float32)
        
    def __len__(self):
        return self.num


# In[4]:


class BottleneckX(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=4):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes * 2, planes // 8, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 8, planes * 2, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu(res)

        return res


class SEResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=375):
        self.inplanes = 64
        super(SEResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((6, 4))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# In[5]:


BATCH_SIZE = 16
NUM_TRAIN = len(train_pairs)
NUM_TEST = len(test_pairs)

trainDataset = COREL_5K(train_pairs, NUM_TRAIN)
train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

valDataset = COREL_5K(val_pairs, NUM_TEST)
val_loader = DataLoader(dataset=valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

testDataset = COREL_5K(test_pairs, NUM_TEST)
test_loader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)


# In[6]:


LEARNING_RATE = 0.001

model = SEResNeXt(BottleneckX, [3, 4, 6, 3], num_classes=374)
model.cuda()
critrien = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[9]:


NUM_EPOCHS = 10
best_acc = 0
for epoch in range(NUM_EPOCHS):
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    for i, (data, label) in tqdm(enumerate(train_loader), total=NUM_TRAIN // BATCH_SIZE, ncols=50, leave=False, unit='b'):
        data = Variable(data).cuda()
        label = Variable(label).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = critrien(output, label)
        train_loss += loss.data[0]
        _, predict = torch.max(output, 1)
        label = label.cpu().data.numpy()
        pred = predict.data
        for i in range(len(pred)):
            if pred[i] in list(np.where(label[i]==1)[0]):
                train_acc += 1
        loss.backward()
        optimizer.step()
        
    for i, (data, label) in enumerate(val_loader):
        data = Variable(data).cuda()
        label = Variable(label).cuda()
        output = model(data)
        loss = critrien(output, label)
        test_loss += loss.data[0]
        _, predict = torch.max(output, 1)
        label = label.cpu().data.numpy()
        pred = predict.data
        for i in range(len(pred)):
            if pred[i] in list(np.where(label[i]==1)[0]):
                test_acc += 1
    
    print('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f'
            %(epoch+1, NUM_EPOCHS, 
              train_loss / NUM_TRAIN, train_acc / NUM_TRAIN, 
              test_loss / NUM_TEST, test_acc / NUM_TEST))
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "models/SEResNext.pkl")

