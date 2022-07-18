import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
from collections import OrderedDict
from numpy import savetxt
from transfer_trainer import Transfer_Trainer
from trainer import Trainer_Tester  

#Define Normalization Parameters#
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#Define Hyperparameters#
batch_size=20
lr=0.01
momentum=0.6
step_size=1
gamma=0.1
num_epochs=3
dropout_ratio=0.25

#Gets Dataset#
data_transforms = {
    'training': transforms.Compose([
        transforms.Resize((512, 512)),               #Resizing
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),           #Normalization
    'validation': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['training', 'validation']}

#Define Dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['training', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation']}
class_names = image_datasets['training'].classes

#--To use CUDA compatible GPU is being checking--#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define Pretrained Resnet18
model_conv = torchvision.models.resnet18(pretrained=True)

#Freeze all parameters
for param in model_conv.parameters():
    param.requires_grad = False

        
#Define FC Layers and add them to end of the Resnet18 instead of last fc layer
#Layers defined newly have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, 200)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=dropout_ratio)),
                                        ('fc2', nn.Linear(200, 11))]))

       
model_conv.fc = classifier
#Model to CUDA 
model_conv = model_conv.to(device)
#Define Loss Function
criterion = nn.CrossEntropyLoss()
#Define Optimization Function
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr, momentum)

# Decay LR by a factor of "gamma" every "step_size" epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size, gamma)

#Define databases to analysis
data_base_training=[] 
data_base_validation=[] 

#Train the model
model, data_base_training, data_base_validation = Transfer_Trainer().train_model(dataloaders,
                                   model_conv, 
                                   criterion, 
                                   optimizer_conv,
                                   exp_lr_scheduler,
                                   data_base_training,
                                   data_base_validation, 
                                   num_epochs,
                                   dataset_sizes)
#Record the findings
data_current_trainig = np.array(data_base_training)  
data_current_validation = np.array(data_base_validation)  
name=f"batch_size:{batch_size}_\
    {lr}_\
    {num_epochs}_\
    {momentum}"    
savetxt(f'trainig_{name}.csv', data_current_trainig, delimiter=',')
savetxt(f'test_{name}.csv', data_current_validation, delimiter=',')

    
     
Trainer_Tester().confusion(image_datasets["validation"],model_conv,class_names)














