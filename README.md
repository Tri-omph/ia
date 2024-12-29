# ia
L'intelligence artificielle en clustering pour les déchets

# Conversion

## Contexte

- Sous Windows, les conversions de `.pth` vers `.onnx` puis vers TensorFlow se sont bien déroulées. Cependant, il semble qu'un des modules nécessaires à la conversion de TensorFlow vers TensorFlow.js ne soit pas utilisable sous Windows, mais uniquement disponible sous Linux.

- Sous Linux, la conversion de TensorFlow vers TensorFlowJS, s'est bien déroulée !

C'est loin d'être une solution admirable, mais cela fonctionne !

## Démarche sous Windows

1. Cloner le projet
2. Créer un environnement virtuel
```bash
virtualenv --python=python3.10 .convert-ia
```

3. Activer l'environnement virtuel
```bash
.convert-ia\Scripts\activate
```
3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Démarche sous Linux

1. Cloner le projet
2. Créer un environnement virtuel
```bash
python -m venv .convert-ia
```
3. Activer l'environnement virtuel
```bash
source .convert-ia/bin/activate
```
3. Installer les dépendances
```bash
pip install -r requirements2.txt
```