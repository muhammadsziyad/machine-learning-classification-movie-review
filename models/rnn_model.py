# models/rnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn_model(input_shape=(200,), max_features=10000):
    model = models.Sequential([
        layers.Embedding(input_dim=max_features, output_dim=128, input_length=input_shape[0]),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Binary cross-entropy for binary classification
                  metrics=['accuracy'])
    return model
