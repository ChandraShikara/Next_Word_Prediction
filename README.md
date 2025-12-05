Next Word Prediction with Recurrent Neural Networks (RNN/LSTM)

This project implements a deep learning model using a Bi-directional Long Short-Term Memory (Bi-LSTM) architecture to predict the next most probable word in a sequence. The model is trained on a literary corpus to capture complex language patterns and contextual dependencies.

🌟 Project Overview

The goal of this project is to create an intelligent text generation system that can accurately predict the next word given a preceding sequence of words. This technology is the core mechanism behind features like autocomplete, predictive text, and basic language modeling.

Key Technology: Bi-directional LSTM (Long Short-Term Memory) - a type of Recurrent Neural Network designed to overcome the limitations of standard RNNs in capturing long-term dependencies.

🛠️ Technical Details

Model Architecture

The model architecture is sequential and is designed for high accuracy in sequence prediction tasks:

Input Layer: A Tokenization layer converts the raw text into numerical sequences.

Embedding Layer: Converts the integer-encoded words into dense, fixed-size vectors. This layer learns a semantic representation of each word.

Bi-directional LSTM Layer: The core of the model. It processes the input sequence both forward and backward, allowing the model to leverage context from both the past and the future of the sequence before making a prediction for the next word.

Dense Layers: Used for feature combination and transformation.

Output Layer: A final Dense layer with a Softmax Activation function. Softmax ensures the output is a probability distribution over the entire vocabulary, indicating the likelihood of each word being the next word. 

Training Metrics

Metric                  Value

Training Corpus    Sherlock Holmes Text (sherlock_holmes_corpus.txt)

Final Accuracy     87.37%

Model Type         Bi-directional LSTM

📁 Repository Contents

File

Description

Next_Word_Prediction.ipynb : The Jupyter Notebook containing the full code for data preprocessing, model creation, training, evaluation, and final prediction testing.

sherlock_holmes_corpus.txt : The raw text file used for training the model (sourced from Project Gutenberg).

next_word_predictor.h5 : The trained Keras model file, containing the architecture, weights, and optimizer state. This file allows for immediate inference without retraining.


🚀 Getting Started

Prerequisites

You will need the following Python libraries installed:

pip install tensorflow numpy

1. Load and Run the Notebook

The easiest way to replicate the results and test the model is to run the cells in the Next_Word_Prediction.ipynb notebook in sequence.

2. Testing Predictions

The model can be loaded directly for making predictions on new seed text. Below is the general workflow demonstrated in the notebook:

Run the notebook to have a model....

import numpy as np

# [Tokenizer loading/initialization code here] 
# Note: The tokenizer object must be loaded or recreated with the same vocabulary used for training.

def predict_next_word(seed_text, n_words_to_generate):
    # 1. Preprocess seed_text (lowercase, tokenize)
    # 2. Pad sequence to match training input length
    # 3. Predict using model.predict(padded_sequence)
    # 4. Convert the highest probability index back to a word
    # 5. Loop n_words_to_generate times
    pass

# Example Usage
seed_text_1 = "my dear watson"
print(f"Input: '{seed_text_1}'")
# Output will be the generated sequence of words


📈 Future Enhancements

Larger Corpus: Train the model on a much larger, more diverse dataset (e.g., Wikipedia) for more generalized predictions.

Hyperparameter Tuning: Systematically tune parameters (e.g., number of LSTM units, dropout rate, sequence length) to maximize accuracy.

Advanced Models: Transition to Transformer-based architectures (like a simplified BERT or GPT) for state-of-the-art performance in long-range context handling.
