# model_setup.py
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define actions
actions = np.array(['hello', 'workout', 'fly'])

# Create and compile the model
def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    model.save('model.h5')

# Load the trained model
def load_trained_model():
    return load_model('model.h5')

# Prepare data
def prepare_data():
    DATA_PATH = os.path.join('MP_Data')

    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(30):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return train_test_split(X, y, test_size=0.05)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
    return multilabel_confusion_matrix(ytrue, yhat), accuracy_score(ytrue, yhat)
