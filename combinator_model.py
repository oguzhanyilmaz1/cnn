import torch.nn as nn
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self,
                    img_size=(124,124),
                    lr=0.05,
                    conv_number=3,
                    fn_number=2,
                    regularization="NoReg",
                    filter_group=[5,5,5,5],
                    activation_function="ReLU",
                    loss_function="NLLLoss",
                    optimizer="SGD",
                    momentum=0.3):
        
        super().__init__()
        #--Parameters are defined--#
        self.lr=lr
        self.conv_number=conv_number
        self.fn_number=fn_number
        self.regularization=regularization
        self.filter_group=filter_group                                         #Size of Kernel for each Conv. Layer Sequentially
        self.momentum=momentum
        self.loss_function=loss_function
        self.last_layer=11                                                     #Number of class
        #--Parameters are defined--#
        
        #--Activation Function is defined--#     
        if activation_function=="ReLU":
            self.act_function=nn.ReLU()
        elif activation_function=="ELU":
             self.act_function=nn.ELU()
        #--Activation Function is defined--# 

        
# %%    #--Convolutional Layers are defined--#   
        
        if conv_number==3:
            self.convolutional_layer= nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=8,kernel_size=filter_group[0]),
                self.act_function,
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=8,out_channels=16,kernel_size=filter_group[1]),
                self.act_function,
                nn.MaxPool2d(2, 2),       
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=filter_group[2]),
                self.act_function,
                nn.MaxPool2d(2, 2)
                )

        elif conv_number==4:
            self.convolutional_layer= nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=8,kernel_size=filter_group[0]),
                self.act_function,
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=8,out_channels=16,kernel_size=filter_group[1]),
                self.act_function,
                nn.MaxPool2d(2, 2),       
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=filter_group[2]),
                self.act_function,
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=filter_group[3]),
                self.act_function,
                nn.MaxPool2d(2, 2)
                )
            self.pool = nn.MaxPool2d(2, 2)
            #--Convolutional Layers are defined--# 
             
# %%        #--Fully Connected Layers are defined--#
         
        
        if fn_number==2 and conv_number==3:
            initial_size=(((img_size[0]-filter_group[0]+1)                     #initial_size is size of first layer that connected Conv Layers
               /2-filter_group[1]+1)
                /2-filter_group[2]+1)/2
            initial_dim=int(initial_size*initial_size*32)
            second_dim=int(initial_dim**0.5)
            
            self.linear_layer = nn.Sequential(
            nn.Linear(in_features=initial_dim, out_features=second_dim),
            self.act_function,
            nn.Linear(in_features=second_dim, out_features=self.last_layer),
        )

        elif fn_number==2 and conv_number==4:

           initial_size=((((img_size[0]-filter_group[0]+1)
               /2-filter_group[1]+1)
                /2-filter_group[2]+1)
                 /2-filter_group[3]+1)/2
           initial_dim=int(initial_size*initial_size*64)
           second_dim=int(initial_dim**0.5)
           
           self.linear_layer = nn.Sequential(
            nn.Linear(in_features=initial_dim, out_features=second_dim),
            self.act_function,
            nn.Linear(in_features=second_dim, out_features=self.last_layer),
        )
        
        elif fn_number==3 and conv_number==3:
            initial_size=(((img_size[0]-filter_group[0]+1)
               /2-filter_group[1]+1)
                /2-filter_group[2]+1)/2
            initial_dim=int(initial_size*initial_size*32)  
            second_dim=int(initial_dim**0.5)
            third_dim=second_dim/2
            
            self.linear_layer = nn.Sequential(
            nn.Linear(in_features=initial_dim, out_features=second_dim),
            self.act_function,
            nn.Linear(in_features=second_dim, out_features=third_dim),
            self.act_function,
            nn.Linear(in_features=third_dim, out_features=self.last_layer),
        )

        
        elif fn_number==3 and conv_number==4:
            initial_size=((((img_size[0]-filter_group[0]+1)
               /2-filter_group[1]+1)
                /2-filter_group[2]+1)
                 /2-filter_group[3]+1)/2
            initial_dim=int(initial_size*initial_size*64)
            second_dim=int(initial_dim**0.5)
            third_dim=second_dim/2
            
            self.linear_layer = nn.Sequential(
            nn.Linear(in_features=initial_dim, out_features=second_dim),
            self.act_function,
            nn.Linear(in_features=second_dim, out_features=third_dim),
            self.act_function,
            nn.Linear(in_features=third_dim, out_features=self.last_layer),
        )
            #--Fully Connected Layers are defined--#
# %%        #--Hyperparameters  are defined--#

        #--Loss Function is defined--#
        if loss_function=="NLLLoss":
              self.criterion = nn.NLLLoss()  
        elif loss_function=="CrossEntropyLoss":
              self.criterion = nn.CrossEntropyLoss()
        #--Loss Function is defined--#   
        
        #--LogSoftmax Function is defined--#         
        if loss_function=="NLLLoss":
             self.softmax_layer = nn.Sequential(
                 nn.LogSoftmax(dim=1))
        #--LogSoftmax Function is defined--#
# %%          
        #--Optimizer and Regularization Method are defined--#  
        if optimizer=="adam":
            if regularization=="L2":  
             self.optimizer = optim.Adam(self.parameters(), lr=lr,weight_decay=1e-5) 
            else:
             self.optimizer = optim.Adam(self.parameters(), lr=lr) 
            
        elif optimizer=="SGD":
            
            if regularization=="L2":  
                self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=self.momentum,weight_decay=1e-5)
            else:
                self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=self.momentum) 
        #--Optimizer and Regularization Method are defined--#  
# %%                
   
    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        if self.loss_function=="NLLLoss":
            x = self.softmax_layer(x)
            
        return x         