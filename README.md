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

## Démarche générale

Sous Windows
1. Démarrer [l'environnement virtuel](#démarche-sous-windows).
2. Lancer les conversions avec les commandes suivantes.

```bash
cd ./conversion
python ./main.py
```

Passer sous Linux
1. Démarrer [l'environnement virtuel](#démarche-sous-linux).
2. Lancer la dernière conversion avec les commandes suivantes.

```bash
cd ./conversion
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --saved_model_tags=serve ./version/v1/IA-v1_tf ./version/v1/IA-v1_tfjs
```
`NB`: Adapter les versions dans les commandes ci-dessus !

3. Copier-coller le dossier obtenu dans votre application


# Serveur Flask

## Contexte

- Après la conversion du modèle en ONNX, on peut créer un serveur Flask pour connecter un fichier Python à l'application.

## Utilisation

- Le serveur Flask est sur le fichier app.py. Le fichier tourne sur l'adresse localhost:5000/predict. Le body de la requête est un form-data de forme key="file" / value=chemin_image.extension (selon l'image). La réponse de la requête est le nom de la classe de l'image (nom en français).

## Démarche sous Windows

```bash
cd ./conversion
pip install requirements3.txt
./app.py
```
