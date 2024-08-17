# Poem -Baudelaire Text Generation using LSTM with TensorFlow
# README for Text Generation using LSTM with TensorFlow

## Overview
This project demonstrates the implementation of a text generation model using LSTM (Long Short-Term Memory) networks with the Keras API in TensorFlow. The dataset used for training is derived from *Les Fleurs du mal* by Charles Baudelaire, which is a collection of poems. The model learns to predict the next word in a sequence, which can then be used to generate poetry or prose similar to the training data.

## Project Structure
The project is structured into the following steps:

1. **Load Packages**: Import necessary libraries and frameworks including TensorFlow, Keras, and others for data processing and visualization.
2. **Download Dataset**: Download the dataset containing *Les Fleurs du mal* if it is not already present.
3. **Data Preprocessing**: 
   - Read the dataset and process it by converting to lowercase, removing special characters, and tokenizing.
   - Prepare sequences of words to train the model on the relationship between these sequences.
   - Apply padding to ensure uniform input sizes for the model.
4. **Model Construction**: 
   - Build a sequential neural network model with an embedding layer, bidirectional LSTM layers, dropout for regularization, and dense layers for output.
5. **Training the Model**: Train the model with the processed data over multiple epochs to learn word sequences.
6. **Text Generation**: Use the trained model to generate new text sequences by predicting the next word given an input sequence.
7. **Visualization**: Plot training accuracy and loss to monitor model performance.
8. **Advanced Text Generation**: Incorporate temperature sampling to add creativity and variability to the generated text.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras (included in TensorFlow 2.x)
- NumPy
- Matplotlib

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib


Utilisation
1. Charger et Prétraiter les Données

Exécutez le script suivant pour télécharger et prétraiter les données textuelles :

python

!wget -nc "http://cedric.cnam.fr/~thomen/cours/US330X/fleurs_mal.txt"

with open("/content/fleurs_mal.txt", 'r', encoding = "utf-8") as f:
    parag = f.readlines()
parag = " ".join(parag[239:]).lower().replace('\n', ' ').replace('\r', '')

2. Construire et Entraîner le Modèle

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

3. Générer du Texte

Générez du texte à partir d'une phrase de départ :

python

initial_text = "Oh toi, mon âme"
total_word_gen = 500

poem = generate_poem(model, initial_text, tokenizer, total_word_gen)
print(poem)

4. Génération Avancée avec l'Échantillonnage par Température

Utilisez l'échantillonnage par température pour une sortie plus variée :

python

poem = generate_poem_with_temperature(model, "Oh toi, mon âme", tokenizer, 500, temperature=0.7)
print(poem)

Résultats

Le modèle génère des séquences de texte qui tentent d'imiter le style et le contenu de la poésie de Baudelaire. Cependant, en raison de la complexité de la langue et du modèle relativement simple, les résultats peuvent varier en qualité, avec des phrases parfois répétitives ou incohérentes.
Remarques

    Augmenter la taille du jeu de données ou améliorer l'architecture du modèle (par exemple, en utilisant des couches LSTM plus profondes) pourrait améliorer la qualité du texte généré.
    Le modèle est entraîné sur un style littéraire spécifique, ce qui le rend moins polyvalent pour d'autres types de génération de texte.

Licence

Ce projet est fourni à des fins éducatives sous la licence MIT.
Conclusion

Ce projet fournit une approche de base pour la génération de texte à l'aide de réseaux de neurones. Il montre le potentiel et les limites de la prédiction de séquences avec des LSTM sur des textes littéraires et peut servir de base pour des expérimentations ultérieures en traitement du langage naturel.

