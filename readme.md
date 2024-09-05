Let's create a TensorFlow project using Keras with the IMDB movie review dataset for sentiment analysis. This project will involve building a simple neural network to classify movie reviews as either positive or negative.

Project Structure
Here’s a typical structure for the project:

```css
tensorflow_imdb/
│
├── data/
│   └── imdb_data.py               # Script to load and preprocess IMDB data
├── models/
│   ├── rnn_model.py               # Script to define and compile an RNN model (with LSTM or GRU)
├── train.py                       # Script to train the model
├── evaluate.py                    # Script to evaluate the trained model
└── utils/
    └── plot_history.py            # Script to plot training history
```


Step 1: Load and Preprocess IMDB Data
Create a file named imdb_data.py in the data/ directory. This script will handle loading and preprocessing the IMDB dataset. We will also pad the sequences to ensure uniform length for the reviews.

```python
# data/imdb_data.py

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(max_features=10000, max_len=200):
    # Load IMDB dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)

    # Pad sequences (each review to have the same length)
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    return (x_train, y_train), (x_test, y_test)
```

Step 2: Define the RNN Model
We'll define a simple RNN model with an LSTM layer in the rnn_model.py file inside the models/ directory.

```python
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
```

Step 3: Train the Model
Create a train.py script at the root of the project to load data, build the model, and train it.

```python
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
model.save('rnn_imdb_model.h5')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
```

Step 4: Evaluate the Model
Create an evaluate.py script to evaluate the trained model on the test data.

```python
# evaluate.py

import tensorflow as tf
from data.imdb_data import load_data

# Load IMDB data
(x_train, y_train), (x_test, y_test) = load_data()

# Load the trained model
model = tf.keras.models.load_model('rnn_imdb_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Step 5: Plot Training History
Use the plot_history.py script to visualize the training and validation accuracy and loss.

```python
# utils/plot_history.py

import matplotlib.pyplot as plt
import pickle

def plot_history(history_file='history.pkl'):
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    plot_history()
```

Step 6: Run the Project
Train the Model: Run the train.py script to start training the model.

```bash
python train.py
```

Evaluate the Model: After training, evaluate the model's performance using evaluate.py.

```bash
python evaluate.py
```

Plot the Training History: Visualize the training history using plot_history.py.

Summary
This project demonstrates how to use the IMDB movie review dataset with TensorFlow and Keras to perform binary sentiment classification using a Recurrent Neural Network (RNN) with an LSTM layer. The model processes sequences of words from the reviews, and we use padding to ensure all sequences are the same length.