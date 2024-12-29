import torch

import torch
from efficientnet_pytorch import EfficientNet
from torchvision import models

def load_model(nomIA, num_classes, usingResnet=False):
    # Recréer le modèle
    if usingResnet:
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        num_features = model._fc.in_features
        model._fc = torch.nn.Linear(num_features, num_classes)

    # Charger les poids
    model.load_state_dict(torch.load(f'{nomIA}.pth'))
    model.eval()  # Mettre le modèle en mode évaluation
    return model

# *************************** conversion pth => onnx
nomIA = "IA-v1"
num_classes = 8
usingResnet = False

model = load_model(nomIA, num_classes)

# Export ONNX
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    f"{nomIA}.onnx",
    opset_version=11,
    input_names=["img_input"],
    output_names=["class_nb_output"],
    dynamic_axes={
        "img_input": {0: "batch_size", 2: "height", 3: "width"},
        "classification_nbs_output": {0: "batch_size"}
    }
)

print("[OK]*************************** conversion pth => onnx\n\n")
# *************************** conversion onnx => tensorflow

from onnx_tf.backend import prepare
import onnx

# Charger le modèle ONNX
onnx_model = onnx.load(f"{nomIA}.onnx")

# Convertir en modèle TensorFlow
tf_rep = prepare(onnx_model)

# Sauvegarder le modèle TensorFlow
tf_rep.export_graph(f"{nomIA}_tf")


print("[OK]*************************** conversion onnx => tensorflow\n\n")
# *************************** conversion tensorflow => tensorflowjs


print("Utilise la commande directement depuis le terminal pour convertir le model")

