import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Reshape data for CNN (assuming each sample has 42 landmarks, for example)
# Adjust the shape to match your data format.
data = data.reshape(-1, 21, 2, 1)  # Example shape (adjust if needed)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# List to store accuracy results for each model
accuracies = {}

### CNN Model ###
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(21, 2, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')  # output layer
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
cnn_accuracy = history.history['val_accuracy'][-1] * 100
accuracies['CNN'] = cnn_accuracy

### Random Forest Model ###
rf_model = RandomForestClassifier()
rf_model.fit(x_train.reshape(len(x_train), -1), y_train)
rf_y_predict = rf_model.predict(x_test.reshape(len(x_test), -1))
rf_accuracy = accuracy_score(rf_y_predict, y_test) * 100
accuracies['Random Forest'] = rf_accuracy

### SVM Model ###
svm_model = SVC()
svm_model.fit(x_train.reshape(len(x_train), -1), y_train)
svm_y_predict = svm_model.predict(x_test.reshape(len(x_test), -1))
svm_accuracy = accuracy_score(svm_y_predict, y_test) * 100
accuracies['SVM'] = svm_accuracy

# Save CNN model
cnn_model.save('cnn_model.h5')
with open('rf_model.p', 'wb') as f:
    pickle.dump({'model': rf_model}, f)

### Plotting Comparison Graph ###
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.show()
