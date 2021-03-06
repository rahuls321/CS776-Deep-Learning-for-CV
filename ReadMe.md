Assignment 1

## Build Neural Network From Scratch on CIFAR-10 Datasets

Following are the package dependencies with their versions 

Packages dependencies

```pip
certifi==2021.10.8
cycler==0.11.0
fonttools==4.29.0
kiwisolver==1.3.2
matplotlib==3.5.1
numpy==1.22.1
opencv-python==4.5.5.62
packaging==21.3
Pillow==9.0.0
pyparsing==3.0.7
python-dateutil==2.8.2
six==1.16.0
torch==1.10.1+cu113
torchaudio==0.10.1+cu113
torchvision==0.11.2+cu113
typing_extensions==4.0.1
```

I have also created one requirements.txt
So do 

```pip
pip install -r requirements.txt
``` 

will install all packages.

This folder contains following directories and files 
1. **cifar-10-batches-py** - contains all the data used in this assignment 1
2. **out_folder** - contains all the output generated by the code including all the plots
    a. Augmented - contains all the plots when model was running on augmented datasets.
    a. Unaugmented - contains all the plots when model was running on unaugmented datasets.
3. **main.py** - The main file for this assignment contains all the code associated with assignment
4. **mlp.py** - It contains the MLP model class with backproagation and other associated functions.
5. **feature_extractor.py** - generate feature vector of an image using ResNet model
6. **run.sh** - script file to run the complete assignment
7. **requirements.txt** - contains the package dependencies
8. **utils.py** - contains all the required functions used in the main.py including preprocessing or generating features function.
9. **model_weights** - contains the pretrained model weights
10. **feat_vector** - contains the feature vector of the generated features from the image of ResNet Block
11. **log** - folder contains all the logs trained with different sets of hyperparameters.
12. **report.pdf** - a report that has all details about the implementation and testing of the model.

`run.sh` is the top-level script that runs the entire assignment.

To run the entire assignment, go to home directory where this README file is there and use --> ./run.sh 

For datasets - Download CIFAR-10 datasets from this link https://www.cs.toronto.edu/~kriz/cifar.html and rename the folder as cifar-10-batches-py

All the hyperparameters are defined in the run.sh file if you want you can tune it from there for training.
If you want to use my pretrained model weights ---- put isModelWeightsAvailable = 1 (defined in the run.sh file)
or test on some other weights, put that weights file(in numpy format) in the model_weights folder and then put isModelWeightsAvailable = 1 and run the ./run.sh command

If you want to use my saved feature vectors ---- pass -f "feat_vec_folder_path" (defined in the run.sh file) (Because it takes time to process)
if not you can remove it from the arguments, it will generate the feature vectors.

File size is so big so I'm providing the drive link of saved weights https://drive.google.com/drive/folders/1RjkzPHDjt54HvwchMHNBhQXU6EM-rXuh?usp=sharing. Download all the feat_vectors and put it in the feat_vector folder and pass the feat_vector folder path and done. 

### ./run.sh -- this will generate all the results in the out_folder. 
