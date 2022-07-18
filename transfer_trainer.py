import torch
import time
import copy

class Transfer_Trainer:
     def __init__(self):
         pass
     def train_model(self,dataloaders,model, criterion, optimizer, 
                    scheduler,data_base_training,
                    data_base_validation, num_epochs,dataset_sizes):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #Record time
        since = time.time()
        #Get copy of modelstate
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['training', 'validation']:
                if phase == 'training':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'training'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'training':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                if phase == 'training':
                    scheduler.step()
                #Gets loss
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                #Add to databases
                if phase == 'training':
                    data_base_training.append([epoch,epoch_acc,epoch_loss]) 
                else:    
                    data_base_validation.append([epoch,epoch_acc,epoch_loss]) 
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, data_base_training, data_base_validation
            
   
    