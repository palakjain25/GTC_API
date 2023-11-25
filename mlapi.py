#libraries
from fastapi import FastAPI
from pydantic import BaseModel
import ast
import pandas as pd
import numpy as np
import random
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

app=FastAPI()

class ScoringItem(BaseModel):
    input_genre: str
    input_chord_sequence: int
    input_creativity: int

model = load_model('GTC.h5')
dataset= pd.read_csv('GTC_API_Dataset.csv')

text_genre = list(dataset['genres'].values)
words_to_skip = [word for sublist in [genre.split() for genres_list in text_genre for genre in ast.literal_eval(genres_list)] for word in sublist]

unique_word_to_skip=np.unique(words_to_skip)
new_element = 'post'
unique_word_to_skip = np.append(unique_word_to_skip, new_element)
new_element = 'r'
unique_word_to_skip = np.append(unique_word_to_skip, new_element)
new_element = 'singer'
unique_word_to_skip = np.append(unique_word_to_skip, new_element)


text=list(dataset.combined_list.values)
joined_text= " ".join(text)
partial_chords=joined_text[:10000000]
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_chords)
unique_tokens=np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
n_words=1
input_words = []
next_words = []

for i in range(len(tokens)-n_words):
  input_words.append(tokens[i:i + n_words])
  next_words.append(tokens[i+n_words])
  
def predict_next_chord(input_text, n_best):
    global unique_word_to_skip  # Access the global variable

    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1

    predictions = model.predict(X)[0]

    # Set probabilities of words to be skipped to zero
    for word in unique_word_to_skip:
        if word in unique_token_index:
            predictions[unique_token_index[word]] = 0

    # Get the top n_best predictions after excluding words to be skipped
    filtered_predictions = np.argpartition(predictions, -n_best)[-n_best:]

    return filtered_predictions.tolist()

chord_match_list = [
    "C", "D", "E", "F", "G", "A", "B",
    "Cm", "Dm", "Em", "Fm", "Gm", "Am", "Bm",
    "Db", "Eb", "Gb", "Ab", "Bb",
    "Dbm", "Ebm", "Gbm", "Abm", "Bbm"
]

def generate_chord(input_text, text_length, creativity=3):
    global unique_word_to_skip, chord_match_list  # Access the global variables

    word_sequence = input_text.split()
    current = 0
    used_choices = set()  # Keep track of used choices to ensure uniqueness

    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence))[current:current + n_words])
        try:
            # Skip words in unique_word_skip while generating choices
            possible = predict_next_chord(sub_sequence, creativity)
            filtered_choices = [idx for idx in possible if unique_tokens[idx] not in unique_word_to_skip and unique_tokens[idx] not in used_choices]

            # If there are still choices after filtering, select one; otherwise, choose randomly
            if filtered_choices:
                choice = unique_tokens[random.choice(filtered_choices)]
            else:
                choice = random.choice(unique_tokens)
        except:
            choice = random.choice(unique_tokens)

        # Check if the predicted chord is present in chord_match_list
        predicted_chord = choice
        if predicted_chord in [match_chord for match_chord in chord_match_list]:
            used_choices.add(choice)
        else:
            # If not present, choose a random chord from chord_match_list
            choice = random.choice(chord_match_list)
            used_choices.add(choice)

        word_sequence.append(choice)
        current += 1

    return word_sequence

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    # Access input_genre and input_chord_sequence from the item parameter
    input_genre = item.input_genre
    input_chord_sequence = item.input_chord_sequence
    input_creativity = item.input_creativity
    result = generate_chord(input_genre, input_chord_sequence, input_creativity)
    return result