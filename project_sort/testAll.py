import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from utils import print_time

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transformVal = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def testAll(val_dir: str, usingResnet: bool, num_classes: int = 8):
    val_dataset = datasets.ImageFolder(val_dir, transform=transformVal)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print_time("Loading model...")
    if usingResnet:
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(torch.load('../../trained_nets/recycling_vs_trash_efficientnet_v2_3.pth'))
    model.eval()
    model = model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    print_time("Starting no_grad")
    with torch.no_grad():
        # Note: wrt to trashnet, we skip using .next_batch()
        # And use python's auto-unzip
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            if cuda_available:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(images)
            
            # Multi-class classification
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    print_time("no_grad finished, start formatting metrics...")
    # Convert labels and probabilities to numpy arrays for classification_report
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binarize the labels for multi-class ROC curve
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    
    # Calculate AUC for each class
    roc_auc = {}
    fpr = {}
    tpr = {}
    for i in range(num_classes):  # Loop through each class (3 classes)
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
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

if __name__ == "__main__":
    val_dir = '../../images/final_val'
    val_dir_no_bg = '../../images/final_val_no_bg'
    testAll(val_dir, False)
    testAll(val_dir_no_bg, False)