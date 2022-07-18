import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self,size,filt,lr,momentum,regularization,optimizer):
        super().__init__()
        #initial_size is size of first layer that connected Conv Layers
        self.initial_size=int((((((size-filt+1)/2-filt+1)/2)-filt+1)/2)**2*32) 
        self.optimizer=optimizer             
        self.conv1 = nn.Conv2d(3, 8, filt)                  #Convolutional Layer 1 
        self.pool = nn.MaxPool2d(2, 2)                      #Pooling Layer
        self.conv2 = nn.Conv2d(8, 16, filt)                 #Convolutional Layer 2
        self.pool = nn.MaxPool2d(2, 2)                      #Pooling Layer
        self.conv3 = nn.Conv2d(16, 32, filt)                #Convolutional Layer 3
        self.pool = nn.MaxPool2d(2, 2)                      #Pooling Layer

        self.fc1 = nn.Linear(self.initial_size,200 )        #FC Layer 1
        self.fc2 = nn.Linear(200, 11)                       #FC Layer 2
        self.act_function=nn.ReLU()                         #Activation Function 
        self.dropout = nn.Dropout(p=0.25)                   #Dropout Layer
        self.optimizer = optim.SGD(self.parameters(), lr=lr)#Optimizer
        self.criterion = nn.CrossEntropyLoss()              #Loss Function

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batchs
        x = self.fc1(x)
        x=F.dropout(x)
        x=F.relu(x)
        x = self.fc2(x)
        return x
    
class Processor:
    def __init__(self):
       pass

    def get_all_preds(self,model, loader):
        #--To use CUDA compatible GPU is being checking--#
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        all_preds = torch.tensor([]).to(device)
        all_labels=torch.tensor([]).to(device)
        #--To use CUDA compatible GPU is being checking--#
        
        #Getting all Predictions and Ground Truth Labels
        for data in loader:
            # images, labels = batch
            inputs, label = data[0].to(device), data[1].to(device)
            
            all_labels = torch.cat(
                (all_labels, label)
                ,dim=0
            )
            preds = model(inputs)
            all_preds = torch.cat(
                (all_preds, preds)
                ,dim=0
            )
        return all_preds ,all_labels 











