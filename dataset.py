import os
import cv2
from torchvision import transforms

class Dataset:
    
    def __init__(self):
        
        self.directory=""
        self.categories=["Bread",
                         "Dairy product",
                         "Dessert",
                         "Egg",
                         "Fried food",
                         "Meat",
                         "Noodles-Pasta",
                         "Rice",
                         "Seafood",
                         "Soup",
                         "Vegetable-Fruit"]    
        self.training_dataset=[]
        self.validation_dataset=[]
        self.evaluation_dataset=[]
        self.img_size=(124,124)                     #Image Size
        self.transform = transforms.Compose([       #Image Normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    def taining_data_creator(self):
        for category in self.categories:
            path=os.path.join("training",category)     #Dataset Path
            number=self.categories.index(category)          #Class of Image
            for img in os.listdir(path):
                img_data=cv2.imread(os.path.join(path,img))     #Read Image
                img_data=cv2.resize(img_data, self.img_size)    #Resize Image
                img_data = self.transform(img_data)             #Normalize Image
                self.training_dataset.append([img_data,number]) #Add image to dataset
        return self.training_dataset
    
    def validation_data_creator(self):
        for category in self.categories:
            path=os.path.join("validation",category)
            number=self.categories.index(category)
            for img in os.listdir(path):
                img_data=cv2.imread(os.path.join(path,img))
                img_data=cv2.resize(img_data, self.img_size)
                img_data = self.transform(img_data)
                self.validation_dataset.append([img_data,number])
        return self.validation_dataset
    
    def evaluation_data_creator(self):
        for category in self.categories:
            path=os.path.join("evaluation",category)
            number=self.categories.index(category)
            for img in os.listdir(path):
                img_data=cv2.imread(os.path.join(path,img))
                img_data=cv2.resize(img_data, self.img_size)
                img_data = self.transform(img_data)
                self.evaluation_dataset.append([img_data,number])
        return self.evaluation_dataset
        
dataset_tra=Dataset().taining_data_creator()    #Get Training Dataset
dataset_val=Dataset().validation_data_creator() #Get Validation Dataset
dataset_eva=Dataset().evaluation_data_creator() #Get Evaluation Dataset      
classes= Dataset().categories                   #Get Classes of Dataset
        
        
        
        
        
        
        
        