# cnn
In this study, different architecture combinations will be tried for food classification and the architecture obtained in the test results will be compared with the results of the network trained with the transfer learning method.

---------Welcome------------

Scripts consist of 3 Groups 
Group 1:Combination Part
Group 2:Training the best Architecture Part
Group 3: Transfer Learning Part

Note 1:Each part works with independent main scripts.
Note 2:"trainer.py","dataset.py" are common for Part 1 and 2.
Note 3:The modules required for each part are listed at the end.
Note 4:"training","validation","evaluation" folders must be same folder with scripts or define a new path on "dataset.py" in line 34/45/56.
-----Group 1-----
Combination Part
Group 1 Script 1: combinator_main.py
Group 1 Script 2: combinator_model.py
Group 1 Script 3: trainer.py
Group 1 Script 4: dataset.py

-----Group 2-----
Train the Best Arch. Part
Group 2 Script 1: main_new.py
Group 2 Script 2: model_new.py
Group 2 Script 3: trainer.py
Group 2 Script 4: dataset.py
Group 2 Script 5: plotcm.py

-----Group 3-----
Transfer Function Part
Group 3 Script 1: transfer_main.py


----Group 1 Script 1:combinator_main.py----

---combinator_main.py Duties---
"combinator_main" takes the data from "dataset.py" 
"combinator_main" takes the model from "combinator_model.py"  
"combinator_main" trains the model for each of the given combinations and saves the results in .csv format.
---combinator_main.py Organization---
Part1: Imports required modules
Part2: Checks CUDA
Part3: Takes hyperparameter combinastions
Part4: Combination Matrix is created
Part5: Trainig for each combination and record results

----Group 1 Script 2:combinator_model.py----

---combinator_model.py Duties---
"combinator_model" Creat required model for given combination
---combinator_model.py Organization---
Part1: Activation Function is defined
Part2: Convolutional Layers are defined
Part3: Fully Connected Layers are defined
Part4: Hyperparameters  are defined
Part5: Optimizer and Regularization Method are defined

----Group 1/2 Script 3:trainer.py----

---trainer.py Duties---
"trainer.py" trains the model given with given dataset
"trainer.py" tests the model given with given dataset
"trainer.py" plot confusion matrix ##Just using in "main_new.PY" which is Group2##

----Group 1/2 Script 3:dataset.py----

---dataset.py Duties---
"dataset.py" gets the dataset in given size and normalize images
---dataset.py Organization---
Part1: Classes,size,normalization parameters are defined
Part2: Training Data is defined
Part3: Validation Data is defined
Part4: Evaluation Data is defined

----Group 2 Script 1:main_new.py----
---main_new.py Duties---
"main_new.py" takes the data from "dataset.py" 
"main_new.py" takes the model from "model_new.py"  
"main_new.py" trains and test the best model 
"main_new.py" plot the confusion matrix
---main_new.py Organization---
Part1: Imports required modules
Part2: Checks CUDA
Part3: Takes hyperparameters
Part4: Defines loaders
Part5: Trainig the model
Part6: Tests the model
Part7: Evaluates the model

----Group 2 Script 2:model_new.py----
---model_new.py Duties---
Creat required model for given combination
Define all prediction and Ground truth values
---model_new.py Organization---
class Net
Part1: Convolutional Layers are defined
Part2: Fully Connected Layers are defined
Part3: Hyperparameters  are defined
Part4: Optimizer and Regularization Method are defined
class Processor
Part1: All prediction and Ground truth values

----Group 2 Script 2:plotcm.py----
---plotcm.py Duties---
Plot Confusion Matrix



################################           IMPORT APPENDIX               #############################
For Group 1 Modules:
torch
itertools
numpy
os
cv2

For Group 2 Modules:
torch
itertools
numpy
sklearn.metrics
matplotlib.pyplot
os
cv2

For Group 1 Modules:
torch
numpy
torchvision
time
os
copy


--------------------------------Modules and Versions------------------------
conda 			  4.11.0
Python 			  3.8.3
pytorch                   1.7.1
numpy                     1.18.5
torchvision               0.8.2
opencv                    4.0.1
cudatoolkit               10.1.243
--------------------------------Modules and Versions------------------------
################################            IMPORT APPENDIX                 #############################












