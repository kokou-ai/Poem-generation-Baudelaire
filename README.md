# Poem -Baudelaire Text Generation using LSTM with TensorFlow

![Architecture du modèle LSTM](lstm.png)
## Vue d'ensemble
Ce projet démontre l'implémentation d'un modèle de génération de texte utilisant les réseaux LSTM (Long Short-Term Memory) avec l'API Keras de TensorFlow. Le jeu de données utilisé pour l'entraînement provient de *Les Fleurs du mal* de Charles Baudelaire, un recueil de poèmes. Le modèle apprend à prédire le mot suivant dans une séquence, ce qui peut ensuite être utilisé pour générer de la poésie ou de la prose similaire aux données d'entraînement.

## Structure du Projet
Le projet est structuré en plusieurs étapes :

1. **Chargement des Bibliothèques** : Importation des bibliothèques et frameworks nécessaires, y compris TensorFlow, Keras, et d'autres pour le traitement des données et la visualisation.
2. **Téléchargement du Jeu de Données** : Téléchargement du jeu de données contenant *Les Fleurs du mal* si ce dernier n'est pas déjà présent.
3. **Prétraitement des Données** :
   - Lecture du jeu de données et traitement pour convertir le texte en minuscules, enlever les caractères spéciaux et tokeniser.
   - Préparation des séquences de mots pour entraîner le modèle sur la relation entre ces séquences.
   - Application du padding pour assurer des tailles d'entrée uniformes pour le modèle.
4. **Construction du Modèle** :
   - Construction d'un modèle de réseau de neurones séquentiel avec une couche d'embedding, des couches LSTM bidirectionnelles, un dropout pour la régularisation, et des couches denses pour la sortie.
5. **Entraînement du Modèle** : Entraînement du modèle avec les données traitées sur plusieurs époques pour apprendre les séquences de mots.
6. **Génération de Texte** : Utilisation du modèle entraîné pour générer de nouvelles séquences de texte en prédisant le mot suivant donné une séquence d'entrée.
7. **Visualisation** : Traçage de l'exactitude et de la perte de l'entraînement pour surveiller les performances du modèle.
8. **Génération Avancée de Texte** : Incorporation de l'échantillonnage par température pour ajouter de la créativité et de la variabilité au texte généré.

## Prérequis
- Python 3.x
- TensorFlow 2.x
- Keras (inclus dans TensorFlow 2.x)
- NumPy
- Matplotlib

Installez les dépendances avec :

```bash ```
pip install tensorflow numpy matplotlib


Utilisation
### 1. Charger et Prétraiter les Données

Exécutez le script suivant pour télécharger et prétraiter les données textuelles :

python

!wget -nc "http://cedric.cnam.fr/~thomen/cours/US330X/fleurs_mal.txt"

with open("/content/fleurs_mal.txt", 'r', encoding = "utf-8") as f:
    parag = f.readlines()
parag = " ".join(parag[239:]).lower().replace('\n', ' ').replace('\r', '')

### 2. Construire et Entraîner le Modèle

Pour construire et entraîner le modèle LSTM :

python

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=30, verbose=1)

### 3. Générer du Texte

Générez du texte à partir d'une phrase de départ :

python

initial_text = "Oh toi, mon âme"
total_word_gen = 500

poem = generate_poem(model, initial_text, tokenizer, total_word_gen)
print(poem)

### 4. Génération Avancée avec l'Échantillonnage par Température

Utilisez l'échantillonnage par température pour une sortie plus variée :

python

poem = generate_poem_with_temperature(model, "Oh toi, mon âme", tokenizer, 500, temperature=0.7)
print(poem)

## Résultats

Le modèle génère des séquences de texte qui tentent d'imiter le style et le contenu de la poésie de Baudelaire. Cependant, en raison de la complexité de la langue et du modèle relativement simple, les résultats peuvent varier en qualité, avec des phrases parfois répétitives ou incohérentes.

## Remarques

    Augmenter la taille du jeu de données ou améliorer l'architecture du modèle (par exemple, en utilisant des couches LSTM plus profondes) pourrait améliorer la qualité du texte généré.
    Le modèle est entraîné sur un style littéraire spécifique, ce qui le rend moins polyvalent pour d'autres types de génération de texte.


## Conclusion

Ce projet fournit une approche de base pour la génération de texte à l'aide de réseaux de neurones. Il montre le potentiel et les limites de la prédiction de séquences avec des LSTM sur des textes littéraires et peut servir de base pour des expérimentations ultérieures en traitement du langage naturel.

