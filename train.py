import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from densenet121 import DenseNet
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

def load_data():
    data = []
    labels = []
    random.seed(42)
    image_paths = sorted(list(os.listdir("colon_image_sets")))
    random.shuffle(image_paths)

    for img in image_paths:
        path = sorted(list(os.listdir("colon_image_sets/" + img)))
        for i in path:
            image = cv2.imread("colon_image_sets/" + img + '/' + i)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data.append(image)
            labels.append(img.split("/")[-1])
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    mlb = LabelBinarizer()
    transformed_labels = mlb.fit_transform(labels)

    print("Transformed labels shape:", transformed_labels.shape)
    print("Transformed labels example:", transformed_labels[:5])

    (xtrain, xtest, ytrain, ytest) = train_test_split(data, transformed_labels
                                                      , test_size=0.25, random_state=42)
    return xtrain, xtest, ytrain, ytest


@tf.keras.utils.register_keras_serializable()
class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    @classmethod
    def from_config(cls, config):
        autoencoder = cls(input_shape=(224, 224, 3))
        return autoencoder

def train_densenet(activation_maps_train, activation_maps, history_file_path_densenet, y_train, y_valid):

    model = DenseNet(activation_maps_train.shape[1:])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    history = model.fit(train_datagen.flow(activation_maps_train, y_train, batch_size=32),
                        steps_per_epoch=len(activation_maps_train) // 32,
                        epochs=100,
                        validation_data=(activation_maps, y_valid),
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])

    model.save("model_output_densenet" + "/model.keras")
    with open(history_file_path_densenet, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    loss, accuracy = model.evaluate(activation_maps, y_valid)
    print("history in densenet: ", history.history)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    return loss, accuracy, model


def train_autoencoder(input_shape, x_train, X_valid, history_file_path_autoencoder):

    autoencoder = Autoencoder(input_shape)
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    history = autoencoder.fit(x_train, x_train,
                              epochs=100,
                              batch_size=64,
                              shuffle=True,
                              validation_data=(X_valid, X_valid))
    print("history in autoencoder : ", history.history)

    autoencoder.save("model_output_autoencoder" + "/model.keras")
    with open(history_file_path_autoencoder, 'wb') as history_file:
        pickle.dump(history.history, history_file)


    # autoencoder = tf.keras.models.load_model(os.path.join("model_output_autoencoder_working100", "model.keras"))

    decoded_imgs, activation_maps = autoencoder.predict(X_valid)
    decoded_imgs_train, activation_maps_train = autoencoder.predict(x_train)
    print("activation maps : ", activation_maps.shape)
    channel_index = 0


    plt.figure(figsize=(40, 8))
    for i in range(10):  # Change the range as needed
        plt.subplot(2, 10, i + 1)
        plt.imshow(activation_maps[i, :, :, channel_index],
                   cmap='gray')  # Display the ith activation map for the chosen channel
        plt.title(f'Activation Map {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_valid[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    return activation_maps, activation_maps_train, decoded_imgs


if __name__ == '__main__':


    os.makedirs("model_output_autoencoder", exist_ok=True)
    os.makedirs("model_output_densenet", exist_ok=True)
    history_file_path_autoencoder = os.path.join("model_output_autoencoder", 'training_history.pkl')
    history_file_path_densenet = os.path.join("model_output_densenet", 'training_history.pkl')

    channel = 3
    num_classes = 2
    batch_size = 16
    nb_epoch = 10
    x_train, X_valid, y_train, y_valid = load_data()

    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", X_valid.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_valid:", y_valid.shape)

    x_train = x_train.reshape((len(x_train), 224, 224, 3))
    X_valid = X_valid.reshape((len(X_valid), 224, 224, 3))

    input_shape = x_train.shape[1:]
    activation_maps, activation_maps_train, decoded_imgs = train_autoencoder(input_shape, x_train, X_valid,
                                                                             history_file_path_autoencoder)


    loss, accuracy, model = train_densenet(activation_maps_train, activation_maps, history_file_path_densenet, y_train, y_valid)

    # confusion matrix

    y_pred = model.predict(activation_maps)
    y_pred_classes = np.argmax(y_pred, axis=1)

    auc = roc_auc_score(y_valid, y_pred_classes)
    fpr, tpr, _ = roc_curve(y_valid, y_pred_classes)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    cm = confusion_matrix(y_valid, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    mcc_score = matthews_corrcoef(y_valid, y_pred_classes)
    precision = precision_score(y_valid, y_pred_classes)
    recall = recall_score(y_valid, y_pred_classes)
    f1 = f1_score(y_valid, y_pred_classes)

    print("Accuracy: ",accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Matthews Correlation Coefficient:", mcc_score)

    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
