from tensorflow.keras.datasets import reuters

#load data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#vectorize data to 1-hot encoding
import numpy as np

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train_vec = vectorize_sequences(train_data, dimension=10000)
x_test_vec = vectorize_sequences(test_data, dimension=10000)

y_train_vec = vectorize_sequences(train_labels, dimension=46)
y_test_vec = vectorize_sequences(test_labels, dimension=46)

#create a model
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

#configure compilation
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#create separate validation set
x_val_vec = x_train_vec[:1000]
x_train_partial = x_train_vec[1000:]

y_val_vec = y_train_vec[:1000]
y_train_partial = y_train_vec[1000:]

#train!
history = model.fit(
    x_train_partial,
    y_train_partial,
    epochs=9,
    batch_size=512,
    validation_data=(x_val_vec, y_val_vec)
)

#do whatever you want with the model
