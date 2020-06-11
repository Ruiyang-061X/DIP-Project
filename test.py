from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from densenet_cbam.densenet_cbam import densenet121_cbam
from torch import nn
import torch
import numpy as np


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_testset = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
testset = datasets.ImageFolder('data/test', transform_testset)
testset_loader = DataLoader(testset,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0)         

d = torch.load('checkpoint/densenet121_cbam_data_best_model.pth', map_location='cuda:0')
model = densenet121_cbam()
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 3)
model = model.to(0)
model.load_state_dict(d['state_dict'])

print('test')
print('val acc: {:.4f}'.format(d['best_acc']))
with torch.no_grad():
    model.eval()
    corrects = 0
    corrects0 = 0
    corrects1 = 0
    corrects2 = 0
    for inputs, labels in testset_loader:
        inputs = inputs.to(0)
        labels = labels.to(0)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        corrects0 += ((preds == labels) & (labels == 0)).sum()
        corrects1 += ((preds == labels) & (labels == 1)).sum()
        corrects2 += ((preds == labels) & (labels == 2)).sum()
    acc = float(corrects) / len(testset)
    acc0 = float(corrects0) / (len(testset) / 3)
    acc1 = float(corrects1) / (len(testset) / 3)
    acc2 = float(corrects2) / (len(testset) / 3)
    print('acc: {:.4f}'.format(acc))
    print('acc0: {:.4f}'.format(acc0))
    print('acc1: {:.4f}'.format(acc1))
    print('acc2: {:.4f}'.format(acc2))