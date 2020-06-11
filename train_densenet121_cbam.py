import random
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
from densenet_cbam.densenet_cbam import densenet121_cbam
from torch import nn
from torch import optim


random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_trainset = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
transform_validationset = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
trainset = datasets.ImageFolder('data/train', transform_trainset)
validationset = datasets.ImageFolder('data/val', transform_validationset)
trainset_loader = DataLoader(trainset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=0)
validationset_loader = DataLoader(validationset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)

model = densenet121_cbam(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 3)
model = model.to(0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
num_epochs = 50
best_acc = 0
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)
    print('train')
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in trainset_loader:
        inputs = inputs.to(0)
        labels = labels.to(0)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(trainset)
    epoch_acc = running_corrects.double() / len(trainset)
    print('train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print('validate')
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in validationset_loader:
            inputs = inputs.to(0)
            labels = labels.to(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(validationset)
        epoch_acc = running_corrects.double() / len(validationset)
        print('validate loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))    
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            d = {'state_dict' : model.state_dict(), 'best_acc' : best_acc}
            torch.save(d, 'checkpoint/densenet121_cbam_data_best_model.pth')
        model.train()