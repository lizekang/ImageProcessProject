
# coding: utf-8

# In[22]:


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
from torchvision import transforms
import torchvision.models as models

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from visdom import Visdom
viz = Visdom()
print("visdom: ",viz.check_connection())

ROOT = "Datasets/corel_5k/images/"
dirs = [ROOT+i+"/" for i in next(os.walk(ROOT))[1]]
files = []
[files.extend([i+j for j in next(os.walk(i))[2] if "jpeg" in j]) for i in dirs]

with open("Datasets/corel_5k/labels/training_label") as f:
    train_labels = f.readlines()
with open("Datasets/corel_5k/labels/test_label") as f:
    test_labels = f.readlines()
    train_labels = [i.split(" ")[:] for i in train_labels]
    test_labels = [i.split(" ")[:] for i in test_labels]
    result_label = []
    [result_label.extend([int(j) for j in i if j != '' and j != '\n'][1:])for i in train_labels]
    [result_label.extend([int(j) for j in i if j != '' and j != '\n'][1:])for i in test_labels]
label_dict = {}
for i in result_label:
    if i in label_dict.keys():
        label_dict[i] += 1
    else:
        label_dict[i] = 1
label_index = list(np.argsort(list(label_dict.values()))[-100:])
label_index = list(reversed(label_index))

train_labels = [[int(j) for j in i if j!='' and j != "\n" and int(j)]for i in train_labels]
test_labels = [[int(j) for j in i if j!='' and j != "\n" and int(j)]for i in test_labels]
train_labels = [[j for j in i if i.index(j) == 0 or j in label_index]for i in train_labels]
test_labels = [[j for j in i if i.index(j) == 0 or j in label_index]for i in test_labels]
train_labels = [i for i in train_labels if len(i) > 1]
test_labels = [i for i in test_labels if len(i) > 1]

train_label_dict = {}
for i in train_labels:
    train_label_dict[str(i[0])+".jpeg"] = i[1:]
    
test_label_dict = {}
for i in test_labels:
    test_label_dict[str(i[0])+".jpeg"] = i[1:]


# In[23]:


train_pairs = []
test_pairs = []
for i in files:
    img_name = i.split("/")[-1]
    if img_name in test_label_dict.keys():
        test_pairs.append((i, test_label_dict[img_name]))
    elif img_name in train_label_dict.keys():
        train_pairs.append((i, train_label_dict[img_name]))


# In[48]:


class COREL_5K(Dataset):
    def __init__(self, data, num, trans=None):
        super(COREL_5K, self).__init__()
        self.data = data
        self.num = num
        self.trans = trans
    
    def __getitem__(self, index):
        data_path, label = self.data[index]
        label = [label_index.index(i) for i in label]
        label = np.array(label)
        img = io.imread(data_path)
        if self.trans:
            img = self.trans(img)
        label = np.sum(np.eye(100)[label], axis=0)
        return img, label.astype(np.float32)
        
    def __len__(self):
        return self.num
    
    def _gen_noise_image(self, image, noise_rate):
        noise_image = np.random.uniform(-0.001, 0.001,(image.shape)).astype('float32')
        return noise_rate * noise_image + (1-noise_rate) * image


# In[49]:


BATCH_SIZE = 8
NUM_TRAIN = 100
NUM_TEST = 20

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize([0.3853909028535724, 0.4004333749569167, 0.34717936323577203], [0.2524, 0.2410, 0.2504]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.3853909028535724, 0.4004333749569167, 0.34717936323577203], [0.2524, 0.2410, 0.2504]),
])

trainDataset = COREL_5K(train_pairs, NUM_TRAIN, train_transform)
train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

testDataset = COREL_5K(test_pairs, NUM_TEST, test_transform)
test_loader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)


# In[50]:


a = {}
labels = []
[labels.extend(i[1]) for i in train_pairs]
[labels.extend(i[1]) for i in test_pairs]
for i in labels:
    if i in a.keys():
        a[i] += 1
    else:
        a[i] = 1
for i in a.keys():
    a[i] = 1/a[i]
# weights = torch.FloatTensor(list(a.values())).cuda()


# In[30]:


LEARNING_RATE = 0.001
model = models.resnet50(pretrained=True)
model.avgpool = nn.AvgPool2d(6)
model.fc = nn.Linear(2048, 100)
model.cuda()
critrien = nn.BCEWithLogitsLoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[52]:


# train
NUM_EPOCHS = 10
best_acc = 0
test_gt_dict = {}
for i in list(test_label_dict.values()):
    for j in i:
        if label_index.index(j) in test_gt_dict.keys():
            test_gt_dict[label_index.index(j)] += 1
        else:
            test_gt_dict[label_index.index(j)] = 1

for epoch in range(NUM_EPOCHS):
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    test_pred_dict = {}
    for i in range(100):
        test_pred_dict[i] = 0
    model.train()
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
    model.eval()
    for i, (data, label) in enumerate(test_loader):
        data = Variable(data).cuda()
        label = Variable(label).cuda()
        output = model(data)
        output = torch.nn.functional.sigmoid(output)
        _, predict = torch.sort(output)
        label = label.cpu().data.numpy()
        pred = (predict.data)[:, -4:]
        for i in range(len(pred)):
            same = set(pred[i]) & set(list(np.where(label[i]==1)[0]))
            if len(same):
                for k in same:
                    test_pred_dict[k] += 1
                test_acc += 1
    for i in test_pred_dict.keys():
        if test_pred_dict[i] != 0:
            if test_gt_dict[i] == 0:
                print(i)
            test_pred_dict[i] /= test_gt_dict[i]
    avg_acc = np.mean(list(test_pred_dict.values()))
    print('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Avg Acc: %.4f'
            %(epoch+1, NUM_EPOCHS, 
              train_loss / NUM_TRAIN, train_acc / NUM_TRAIN, 
              test_loss / NUM_TEST, test_acc / NUM_TEST,
              avg_acc))
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "models/ResNext.pkl")

