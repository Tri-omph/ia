import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(image_path, img_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalisation entre 0 et 1
    
    # Normalisation basée sur ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Réorganiser les dimensions si le modèle attend channels_first
    img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    
    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)  # (C, H, W) -> (1, C, H, W)
    
    return img_array




classes = ['Battery', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Textile', 'Trash']

# Charger le modèle TensorFlow
nomIA = "IA-v1"
model_path = f"./version/v1/{nomIA}_tf"
model = tf.saved_model.load(model_path)

# Charger et prétraiter l'image
image_path = '../project_sort/image_test.png'
img = preprocess_image(image_path)

# Convertir en tenseur TensorFlow avec le bon type
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
print(f"Dimension du tensor d'entrée: {img_tensor.shape}")

# Exécuter le modèle avec l'entrée corrigée
output = model.signatures['serving_default'](img_input=img_tensor)

# Afficher le résultat
print(f"Format de la sortie: {output['class_nb_output'].numpy()}")
predicted_class_idx = np.argmax(output['class_nb_output'].numpy())  # Indice de la classe prédite
predicted_class_name = classes[predicted_class_idx]  # Nom de la classe

print(f'The object is classified as: {predicted_class_name}')