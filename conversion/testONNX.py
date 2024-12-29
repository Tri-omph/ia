import onnxruntime as ort
import numpy as np

nomIA = "IA-v1"

# Charger le modèle ONNX
onnx_model_path = f"{nomIA}.onnx"
session = ort.InferenceSession(onnx_model_path)

# Créer une entrée de taille dynamique (par exemple, 300x400)
input_shape = (1, 3, 300, 400)
img_input = np.random.rand(*input_shape).astype(np.float32)

# Effectuer une prédiction
inputs = {"img_input": img_input}
outputs = session.run(["class_nb_output"], inputs)

# Afficher les résultats
print("Output:", outputs[0])
