# train.py

import tensorflow as tf
from data.imdb_data import load_data
from models.rnn_model import build_rnn_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load IMDB data
(x_train, y_train), (x_test, y_test) = load_data()

# Build the RNN model
model = build_rnn_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Train the model
history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    batch_size=64)

# Save the trained model
model.save('rnn_imdb_model.keras')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)