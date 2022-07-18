import torch
from dataset import dataset_tra,dataset_val,dataset_eva                        #Getting Data
from combinator_model import Net                                               #Getting Model
import itertools
import numpy as np
from numpy import savetxt
from trainer import Trainer_Tester                                             #Getting Train Model

#--To use CUDA compatible GPU is being checking--#
CUDA_LAUNCH_BLOCKING=1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#--To use CUDA compatible GPU is being checking--#


#--To tune hyperparameters combinations are being selecting. --#
batch_size=[5,25]                                                              #Number of Batches
lr=[0.05,0.005]                                                                #Learning Rate
conv_number=[3,4]                                                              #Number of Convolutional Layer
fn_number=[2,3]                                                                #Number of Fully Connected Layer
image_normalization=[True]                                                     #Image Normalization Choice
regularization=["NoReg","L2"]                                                  #Regularization Methods
filter_group=[[5,5,5,5]]                                                       #Size of Kernel for Conv. Layers
activation_function=["ReLU","ELU"]                                             #Activation Function
loss_function=["CrossEntropyLoss"]                                             #Loss Function
epoch=[10]                                                                    #Number of Epochs
optimizer=["SGD"]                                                              #Optimizer
momentum=[0.3,0.9]                                                             #Momentum
#--To tune hyperparameters combinations are being selecting. --#

#--The matrix containing the combination of all parameters is obtained. --#
combination_matrix=[batch_size,
                    lr,
                    conv_number,
                    fn_number,
                    image_normalization,
                    regularization,
                    filter_group,
                    activation_function,
                    loss_function,
                    epoch,
                    optimizer,
                    momentum
                    ]
combination_list=list(itertools.product(*combination_matrix))
#--The matrix containing the combination of all parameters is obtained. --#

"""
Training is performed for each architecture in the combination list and the 
results are saved in each iteration in .csv format. The reason for this is not 
to lose the successful trainings in case of an error in an incompatibility that 
may occur in the architecture, and additional ram usage is prevented by deleting 
the current data after each recording.
"""
for i in range(len(combination_list)):
    #--Getting parameters of current architecture --#
    percent=i/256*100
    batch_size=combination_list[i][0]
    lr=combination_list[i][1]
    conv_number=combination_list[i][2]
    fn_number=combination_list[i][3]
    image_normalization=combination_list[i][4]
    regularization=combination_list[i][5]
    filter_group=combination_list[i][6]
    activation_function=combination_list[i][7]
    loss_function=combination_list[i][8]
    epoch=combination_list[i][9]
    optimizer=combination_list[i][10]
    momentum=combination_list[i][11]
    #--Getting parameters of current architecture --# 
    
    #--The remaining data is extracted to comply with the number of batches used. --# 
    remains_tra=len(dataset_tra)%batch_size
    remains_tra=len(dataset_tra)-remains_tra
    dataset_tra=dataset_tra[:remains_tra]
    remains_val=len(dataset_val)%batch_size
    remains_val=len(dataset_val)-remains_val
    dataset_val=dataset_val[:remains_val]
    remains_eva=len(dataset_eva)%batch_size
    remains_eva=len(dataset_eva)-remains_eva
    dataset_eva=dataset_eva[:remains_eva]
    #--The remaining data is extracted to comply with the number of batches used. --# 

    
    #--The loader is determined for each of the training validation and evaluation steps. --# 
    trainloader = torch.utils.data.DataLoader(dataset_tra,  batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                        shuffle=False, num_workers=0)
    evaluationloader = torch.utils.data.DataLoader(dataset_eva,  batch_size=batch_size,
                                            shuffle=True, num_workers=0) 
    #--The loader is determined for each of the training, validation and evaluation steps. --# 
    
    #--Parameters are being sent to the model --# 
    image_size=(124,124)
    net = Net(      image_size,lr,
                    conv_number,
                    fn_number,
                    regularization,
                    filter_group,
                    activation_function,
                    loss_function,
                    optimizer,
                    momentum)
    #--Parameters are being sent to the model --# 
    
    net.to(device)                                                             #Moving model to GPU module if it exists
    data_base=[]                                                               #List to be used for recordings
    lrdc=False                                                                 #Reducing Learning Rate
    
    #--Training in progress --# 
    training_data_base,training_accuracy,loss =Trainer_Tester().training(epoch,
                                                     net,
                                                     lr,
                                                     trainloader,
                                                     data_base,
                                                     len(dataset_tra),
                                                     lrdc)
    #--Training in progress --# 
    test_accuracy=Trainer_Tester().testing(testloader,net,len(dataset_val)) 
    evaluation_accuracy=Trainer_Tester().testing(evaluationloader,net,len(dataset_eva))             
   
    #--Recording the training output --# 
    data_current = np.array(training_data_base)  
    name=f"_batch_size:{batch_size}_\
    {lr}_\
    {conv_number}_\
    {fn_number}_\
    {image_normalization}_\
    {regularization}_\
    {filter_group}_\
    {activation_function}_\
    {loss_function}_\
    {epoch}_\
    {optimizer}_\
    {momentum}_\
    {test_accuracy}_\
    {evaluation_accuracy}_"
    
    savetxt(f'{name}.csv', data_current, delimiter=',')
    del data_current
    #--Recording the training output --# 
    
        












