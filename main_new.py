import torch
from dataset import dataset_tra ,dataset_val,dataset_eva,classes        #Getting Data and Classes
from model_new import Net                                               #Getting Model
from trainer import Trainer_Tester                                      #Getting Train and Test Model

#--To use CUDA compatible GPU is being checking--#
CUDA_LAUNCH_BLOCKING=1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#--To use CUDA compatible GPU is being checking--#

batch_size=25               #Batch Size
image_size=(124,124)        #Image Size
filt=[5,5]                  #Kernel Size
lr=0.05                     #Learning Rate
momentum=0.3                #Momentum
cnv_number=3                #Number of Convolutional Layers
fc=3                        #Number of Fully Connected Layers
regularization="l2"         #Regularization Method "Dropout is including in trainer module" and not stated here
epoch_number=50             #Epoch Number
optimizer="SGD"             #Optimizer

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
testloader = torch.utils.data.DataLoader(dataset_val,  batch_size=batch_size,
                                            shuffle=True, num_workers=0)
evaluationloader = torch.utils.data.DataLoader(dataset_eva,  batch_size=batch_size,
                                            shuffle=True, num_workers=0) 
#--The loader is determined for each of the training validation and evaluation steps. --#
 
#--Parameters are being sent to the model --#
net = Net(image_size[0],
          filt[0],
          lr,
          momentum,
          regularization,
          optimizer
          )
#--Parameters are being sent to the model --#

net.to(device)                                            #Moving model to GPU module if it exists
training_data_base=[]                                     #List to be used for recordings
lrdc=True                                                 #Reducing Learning Rate

#--Training in progress --# 
training_data_base,training_accuracy,loss =Trainer_Tester().training(epoch_number,
                                                     net,
                                                     lr,
                                                     trainloader,
                                                     training_data_base,
                                                     len(dataset_tra),
                                                     lrdc)
#Getting Test and Evaluation Accuracy
test_accuracy=Trainer_Tester().testing(testloader,net,len(dataset_val)) 
evaluation_accuracy=Trainer_Tester().testing(evaluationloader,net,len(dataset_eva))             
print(f"Training accuracy: {training_accuracy}\nValidation accuracy: {test_accuracy}\nEvaluation accuracy: {evaluation_accuracy}")   
#Plot Confusion Matrix
Trainer_Tester().confusion(dataset_eva,net,classes)