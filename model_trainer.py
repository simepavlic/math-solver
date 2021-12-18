import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
import sys


def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='softmax'))
    return model


if __name__ == "__main__":
    class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "plus", "minus",
                   "multiply", "divide", "left_bracket", "right_bracket"]

    data_dir = sys.argv[1]
    X_train, X_test, y_train, y_test = ([] for i in range(4))
    for root, _, files in os.walk(data_dir):
        if len(files) == 0:
            continue
        label = [i for i in range(len(class_names)) if class_names[i] in root][0]
        X = []
        y = [label] * len(files)
        for file in files:
            image = cv2.imread(root + '/' + file)
            image = image.astype('float32')
            image /= 255
            X.append(image)
        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train.extend(X_train_tmp)
        X_test.extend(X_test_tmp)
        y_train.extend(y_train_tmp)
        y_test.extend(y_test_tmp)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    model = cnn_model()

    # since we have categorical target values we are using categorical crossentropy as our loss
    # accuracy is used as a metric because we value prediction accuracy of all classes the same and our data is balanced
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(test_acc)
    model.save('math_model')
