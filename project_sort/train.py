import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import print_time
from efficientnet_pytorch import EfficientNet
from os.path import exists
from tqdm import tqdm

def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # Switch to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():  # No gradients required for validation
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss

def train_model(train_dir: str,
                val_dir: str,
                num_classes: int,
                usingResnet: bool,
                num_epochs: int = 10,
                patience: int = 2) -> tuple[EfficientNet | models.ResNet, DataLoader]:
    """
    Train a model using supervised learning to be able to accurately predict the class of a given image.
    
    
    Parameters
    ----------
    train_dir : str
        The folder location of the training images
    val_dir : str
        The folder location of the validation images
    num_classes : int
        The amount of labels
    usingResnet : bool
        Whether we're using Resnet or EfficientNet as our training model
    num_epochs : int, optional
        The amount of cycles to train for, by default 10
    patience : int, optional
        If the loss does not decrease for <patience> epochs, quit training, by default 2
    
    Returns
    -------
    (EfficientNet | models.ResNet, DataLoader)
        A tuple with the trained model, as well as the loaded validation images
    """
    # Define transformations for training and validation
    # ImageNet mean and std
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

    # Load datasets
    print("Load datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=transform['val'])
    print("Dataset loaded.")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load pretrained ResNet50
    print("Load model...")
    if usingResnet:
        model = models.resnet50()
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    print("Model loaded")

    # Layer freezing
    for param in model.parameters():
        param.requires_grad = False

    if usingResnet:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        optimizer = optim.RMSprop(model.fc.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
    else:
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
        optimizer = optim.RMSprop(model._fc.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
        
    # Weight_decay allows us to slow down near extremum, momentum smooths out noise and avoids local minimas
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    #check_model_parameter_sizes(model)
    
    # Training settings
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    #epoch_loss = []

    ## Training ##
    print_time(f"\n\n\nStarting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        print_time(f"Epoch {epoch+1}:")
        model.train()
        running_loss = 0.0
        iteration = 0

        # Note: wrt to trashnet, we skip using .next_batch()
        # And use python's auto-unzip
        for images, labels in tqdm(train_loader):
            iteration += 1

            images, labels = images.to(device), labels.to(device)
            if cuda_available:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #epoch_loss.append(loss.item())
        
        print_time("Evaluating...")
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print_time(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print_time("Early stopping triggered.")
            break

        # Step the scheduler
        scheduler.step()
        print()
        
    print_time("Finished training")

    return model, val_loader

def save(model, path):
    # Testing takes a while, so save the model before just in case smth bad happens
    ver = 0
    save_path = path+str(ver)+".pth"
    while exists(save_path):
        ver += 1
        save_path = path+str(ver)+".pth"
    print(f"Saved as {save_path}")

    torch.save(model.state_dict(), save_path)
    print_time("Model saved")
    
    return save_path

def test(model, val_loader, num_classes):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    ## Testing ##
    model.eval()

    all_labels = []
    all_probs = []

    print_time("Starting no_grad")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            # Multi-class classification
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    # Convert labels and probabilities to numpy arrays for classification_report
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binarize the labels for multi-class ROC curve
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    
    # Calculate AUC for each class
    roc_auc = {}
    fpr = {}
    tpr = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'blue', 'red', 'purple', 'cyan', 'magenta', 'yellow']
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Each Class')
    plt.legend(loc='lower right')
    plt.show()

    # print_time classification report
    print("Classification Report:")
    print(classification_report(all_labels, np.argmax(all_probs, axis=1), target_names=[f"Class {i}" for i in range(num_classes)]))
    #for i in range(num_classes):
    #    print(f"AUC for Class {i}: {roc_auc[i]:.2f}")
        
    print_time("End of testing")
    return


def check_model_parameter_sizes(model: torch.nn.Module):
        
        # Number of groups of parameters
        print('Number of groups of parameters {}'.format(len(list(model.parameters()))))
        print('-'*55)
        # Print parameters

        total = 0
        for name, param in model.named_parameters():
            print(f"{name:30} : [{param.numel():6}] {param.size()}")
            total += param.numel()

        print(f"{'TOTAL':30} : [{total:6}]")

        print('-'*55)
# https://github.com/garythung/trashnet/tree/master
