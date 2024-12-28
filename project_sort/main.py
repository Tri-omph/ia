import cv2
import os
from sys import argv, exit

from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import randomize_data
import train
import rm_background
import test

# Training or testing
if len(argv) < 2:
    option = ""
    while (option not in ['0', '1']):
        option = input("Please specify whether you'd like to train (0) or test (1) the neural network: ")
else:
    option = argv[1]
    while (option not in ['0', '1']):
        option = input("Please specify whether you'd like to train (0) or test (1) the neural network: ")
option = int(option)

# What model to use/image to test?
if len(argv) < 3:
    if option == 0:
        model_name = ""
    else:
        model_name = "../../trained_nets/recycling_vs_trash_efficientnet_v2_4.pth"
        imageToRead = input("Please specify image path to use. ")
else:
    if option == 0:
        model_name = argv[2]
    else:
        model_name = "./recycling_vs_trash_efficientnet_v2_4.pth"
        imageToRead = argv[2]

# Images not included to reduce bandwidth needed to import
raw_dir = './dataset'
raw_dir_aug = './dataset_aug'
train_dir = './dataset_train'
val_dir = './dataset_test'

model_path = "./recycling_vs_trash_efficientnet_v2_"
#model_name = "../../trained_nets/recycling_vs_trash_efficientnet_v2_5.pth"

final_img_name = "output.jpg"

num_classes = 8
classes = ['Battery', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Textile', 'Trash']

# A very, very annoying difference between ResNet and EfficientNet is accessing their last layer
# ResNet uses .fc, EfficientNet uses ._fc
# Very unnoticable and annoying to debug
usingResNet = False

if __name__ == "__main__":
    if option == 0:
        
        # -1- Augmentation of the dataset (see randomize_data.py)
        assert os.path.exists(raw_dir), "Error. Image dataset does not exist, or is not located at './dataset'. You can download the dataset at https://kaggle.com/datasets/5acfe32544891f206266fb70ca374b5f5fc2f8340b2e9d262947f47e9550fdec."
        if (not os.path.exists(raw_dir_aug)) or (not os.path.exists(train_dir)) or (not os.path.exists(val_dir)):
            print("Warning: This program creates temporary folders to split the dataset into training dataset and test dataset.")
            proceedCreate = input("Would you still like to proceed? (Y/n) ")
            if proceedCreate != "Y":
                exit("Aborting program...")
            
            print("-- Augmentation of dataset, the images are in ", raw_dir_aug)
            randomize_data.clear_folders([train_dir, val_dir, raw_dir_aug])
            randomize_data.augment_data(raw_dir, raw_dir_aug)
            randomize_data.split_data(raw_dir_aug, train_dir, val_dir)
        
        # -2- Model training (see train.py) only if the .pth that represents the model wieghts does not exist
        if os.path.isfile(model_name):
            print("-- Model already trained")

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # Transformations for training and validation
            transform = {
                'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            }
            val_dataset = datasets.ImageFolder(val_dir, transform=transform['val'])
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            num_classes = 8

            if usingResNet:
                model = models.resnet50()
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                model = EfficientNet.from_pretrained('efficientnet-b0')
                model._fc = nn.Linear(model._fc.in_features, num_classes)
            model.load_state_dict(torch.load(model_name, weights_only=True))
        else:
            print("-- Model training")
            model, val_loader = train.train_model(train_dir, val_dir, num_classes, usingResNet)
            model_name = train.save(model, model_path)

        # -3- Showing metrics
        print("\n\n-- Metrics of the model")
        train.test(model, val_loader, num_classes)
        
        print("Deleting tempoary files...")
        randomize_data.clear_folders([train_dir, val_dir, raw_dir_aug])
    elif option == 1:
        # -4- Applying the model to an image defined as an argument
        print("-- Finding class of the image")
        img= cv2.imread(imageToRead)

        rm_background.remove_background(img, final_img_name)
        test.test_1image(final_img_name, model_name, classes, usingResNet)


    