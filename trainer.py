import torch
from model_new import Processor                  #Module that gets all predictions and ground truth values
from plotcm import plot_confusion_matrix         #Module that plot confusion matrix
from sklearn.metrics import confusion_matrix     #Module that get confusion matrix
import matplotlib.pyplot as plt

class Trainer_Tester:
    def __init__(self):
       
        self.processor=Processor()
        
    #--Function that trains the model--#
    def training(self,epoch_number,net,lr,trainloader,training_data_base,dataset_tra_len,lrdc):
        
        #--To use CUDA compatible GPU is being checking--#
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        #--To use CUDA compatible GPU is being checking--#
        
        for epoch in range(epoch_number):                       # loop over the dataset multiple times
            correct=0                                           # number of correct prediction
            running_loss = 0.0                                  # current training loss to record
            running_loss_test= 0.0                              # current training loss to print
            
            if lrdc:                                            # decreasing learning rate
                for g in net.optimizer.param_groups:            # learning rate is reduced every epoch as shown in report
                        lr=lr-((1/(epoch+1))*0.0111)
                        g['lr'] = lr
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                net.optimizer.zero_grad()
                # getting outputs
                outputs = net(inputs)
                # calculate loss
                loss = net.criterion(outputs, labels)
                # graph is differentiated
                loss.backward()
                # optimizer iterated over all parameters
                net.optimizer.step()
                # getting correct classifications
                correct += (torch.squeeze(torch.topk(outputs, 1).indices,1) == labels).float().sum()
                # getting losses
                running_loss += loss.item()
                running_loss_test += loss.item()
                
                if i % 5 == 4:    # print every 5 minibatch
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_test / 5:.12f}')
                    running_loss_test = 0
            # getting accuracy        
            accuracy=float(100*correct/dataset_tra_len)
            # getting total loss for epoch
            loss=running_loss/len(trainloader)
            # recording to database
            training_data_base.append([epoch,accuracy,loss]) 
        return training_data_base,accuracy,loss 
    
    def testing(self,testloader,net,dataset_val_len):
        #--To use CUDA compatible GPU is being checking--#
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        #--To use CUDA compatible GPU is being checking--#
        
        test_accuracy=0
        test_correct = 0
        
        #Gradients eliminated
        with torch.no_grad():
            for data in testloader:
                 # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                 # getting outputs
                outputs = net(inputs)
                # getting correct classifications
                test_correct += (torch.squeeze(torch.topk(outputs, 1).indices,1) == labels).float().sum()
        # getting accuracy  
        test_accuracy=float(100*test_correct/dataset_val_len)
        return test_accuracy
    
    def confusion(self,dataset_eva,net,classes):
        #Gradients eliminated
        with torch.no_grad():
            #Getting all predictions and ground truth values
            prediction_loader = torch.utils.data.DataLoader(dataset_eva, batch_size=5)
            test_preds,all_labels = self.processor.get_all_preds(net, prediction_loader)
            
        #stacking highest prediction for each step and label that given in dataset
        stacked = torch.stack(
            (
                all_labels
                ,test_preds.argmax(dim=1)
            )
            ,dim=1
        )
        
        #creating confusion matrix
        cmt = torch.zeros(11,11, dtype=torch.int64)
        
        #for cell for matrix getting number of true predictions
        for p in stacked:
            tl, pl = p.tolist()
            tl, pl =  int(tl), int(pl)
            cmt[tl, pl] = (cmt[tl, pl]) + 1
            
        cm = confusion_matrix(all_labels.cpu(), test_preds.argmax(dim=1).cpu())    
        plt.figure(figsize=(13,13))    
        plot_confusion_matrix(cm, classes)