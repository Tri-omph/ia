import onnxruntime as ort
import numpy as np
from PIL import Image

def preprocess_image_onnx(image_path, input_size=(224, 224)):
    """
    Prétraite une image pour un modèle ONNX.

    Args:
        image_path (str): Chemin de l'image.
        input_size (tuple): Taille de l'image (hauteur, largeur).

    Returns:
        np.ndarray: Image prétraitée sous forme de tenseur.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)  # input_size = (height, width), resize attend (width, height)
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalisation entre 0 et 1

    # Normalisation basée sur ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Changer le format en (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Nom du modèle et chemin de l'image
nomIA = "IA-v1"
onnx_model_path = f"version/v1/{nomIA}.onnx"
image_path = '../project_sort/image_test.png'

# Charger le modèle ONNX
session = ort.InferenceSession(onnx_model_path)

# Identifier le nom des entrées et sorties du modèle
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prétraiter l'image
img_input = preprocess_image_onnx(image_path)

# Effectuer une prédiction
inputs = {input_name: img_input}
outputs = session.run([output_name], inputs)

# Afficher les résultats
predicted_class_idx = np.argmax(outputs[0])
classes = ['Battery', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Textile', 'Trash']
predicted_class_name = classes[predicted_class_idx]

print("Output:", outputs[0])
print(f'The object is classified as: {predicted_class_name}')
