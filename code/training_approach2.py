# -*- coding: utf-8 -*-
"""final_model.py
    This script follows the convolutional neural network artchitecture for classification of fruits and vegetables
    
    Configuration:
    Python: 3.10.5
    Tensorflow: 2.9.1

    Main points: 5 convolution layers, 2 dropout layers, 2 dense layers, 1 flatten layer, adam optimizer, 20 epochs, early stopping, batch size = 120
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools 
import numpy as np
import absl.logging
import tensorflow as tf
absl.logging.set_verbosity(absl.logging.ERROR)

# Real Image augmentation that generates the batches of tensor data
training_data = ImageDataGenerator(rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Normalizing the data (It will normalize between 0 to 1 raneg instead of 0 to 255)
validation_data = ImageDataGenerator(rescale = 1./255)

# Creating the batches of data with augmentation from directory
train_dir = "/Users/priya/Desktop/final_model/datasets/training"
train_generator = training_data.flow_from_directory(train_dir, 
                                    target_size=(150, 150),
                                    color_mode="rgb", 
                                    class_mode="categorical")

val_dir = "/Users/priya/Desktop/final_model/datasets/testing"
val_gen = validation_data.flow_from_directory(val_dir,
                                    target_size=(150, 150),
                                    color_mode="rgb",
                                    shuffle=False,
                                    class_mode="categorical")

# CNN
cnn_model = tf.keras.models.Sequential([
    # convolution 1
    # Relu is used to make the model non linear
    # max pooling:  To reduce the size of pixels
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(150,150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    # convolution 2
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # convolution 3
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    # convolution 4
    tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    # convolution 5
    tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    # To prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # flatten before feeding into Dense neural network. 
    tf.keras.layers.Flatten(),

    # 512 neurons in the hidden layer
    tf.keras.layers.Dense(512, activation="relu"),

    # To prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Softmax: For multi class classification to activate neurons, output will be 1 for predicted class
    tf.keras.layers.Dense(20, activation="softmax")
]);


cnn_model.summary()

# Saving model artchitecture
# plot_model(cnn_model,to_file="/Users/priya/Desktop/final_model/model2/model_image.png", show_shapes=True,show_layer_names=True)

# categorical_crossentropy loss is used because there will be only one category in output. and using root mean square propagation
cnn_model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

early_stop_call = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
fruit_model = cnn_model.fit(train_generator, batch_size = 120 , epochs=20, validation_data=val_gen, callbacks=early_stop_call)

model_path = "/Users/priya/Desktop/final_model/model2"
tf.keras.models.save_model(
    cnn_model,
    model_path,
    overwrite=True,
    include_optimizer=True,
    save_format="tf",
    signatures=None
)

cnn_model.save("/Users/priya/Desktop/final_model/model2/final_model_modified.h5")

fruit_model.history['accuracy']

# graphs for accuracy
train_acc = fruit_model.history['accuracy'] #training accuracy scores from the model that has been trained
val_acc = fruit_model.history['val_accuracy'] #validation accuracy scores from the model that has been trained
train_loss = fruit_model.history['loss'] #training loss scores from the model that has been trained
val_loss = fruit_model.history['val_loss'] #validation loss scores from the model that has been trained

epochs = range(len(train_acc)) #x axis

plt.plot(epochs, train_acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title('Training and Validation Accuracy Scores')
plt.legend()
plt.figure()

plt.plot(epochs, train_loss, 'ro', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')

plt.title('Training and Validation Loss Scores')

# Displaying the graphs
plt.legend()
plt.show()


cnn_model = tf.keras.models.load_model("/Users/priya/Desktop/final_model/model3_keep_As+old/final_model_modified.h5")
# Confusion matrix

test_score = cnn_model.evaluate(val_gen)
print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100))
print("[INFO] Loss: ",test_score[0])


#plotting the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Confuxion matrix with and without normalization
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

target_names = []
for key in train_generator.class_indices:
    target_names.append(key)

Y_pred = cnn_model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(val_gen.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

#printing accuracy, f1-score, recall and precison
print('Classification Report')
print(classification_report(val_gen.classes, y_pred, target_names=target_names))