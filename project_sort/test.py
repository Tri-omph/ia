import cv2
from sys import argv

from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision import datasets, transforms, models
from utils import print_time

import rm_background

def test_1image(img_path, model_name, classes, usingResNet):

    print_time("Loading validation dataset...")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension (1 image)

    print_time("Loading model...")
    num_classes = len(classes)
    
    if usingResNet:
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model = model.to(device)
    image = image.to(device)

    print_time("Starting inference")
    with torch.no_grad():
        # Forward pass
        outputs = model(image)
        
        # Convert output to probabilities (softmax)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.bar(classes, probs[0], color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities for Each Class')
    plt.xticks(rotation=45)
    plt.show()

    # Output predicted class
    predicted_class = np.argmax(probs)
    print(f"Predicted Class: {predicted_class} (Probability: {probs[0][predicted_class]:.2f})")

    print_time("End of inference")
    

if __name__ == "__main__":
    assert (len(argv) != 1), "You need to specify an image."
    img= cv2.imread(argv[1])

    final_img_name = "output.jpg"
    model_name = "../../trained_nets/recycling_vs_trash_efficientnet_v2_3.pth"
    classes = ['Battery', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Textile', 'Trash']
    
    rm_background.remove_background(img, final_img_name)
    test_1image(argv[1], model_name, classes, False)