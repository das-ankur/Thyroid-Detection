import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_addons as tfa
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


class Model_Finder:
    """
    This class shall be used to find the best model
    """
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    # Build the model
    def model_builder(self):
        model = Sequential()
        # Input layer
        model.add(Input(shape=(25,)))
        # First block
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.5))
        # Second Block
        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(ReLU())
        # Output Layer
        model.add(Dense(4, activation='softmax'))
        print(model.summary)
        # self.logger_object.log(self.file_object, str(model.summary()))
        return model

    def evaluate(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        res = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        print("Evaluation result: ")
        print(res)
        with open(os.path.join("plots", "eval_result.json"), "w") as json_file:
            json.dump(res, json_file)

    def save_loss_and_accuracy(self, history):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # Specify the path and filename for the loss plot image
        loss_plot_path = os.path.join("plots", "loss_plot.png")
        # Save the loss plot as an image
        plt.savefig(loss_plot_path)
        plt.close()
        # Plotting accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # Specify the path and filename for the accuracy plot image
        accuracy_plot_path = os.path.join("plots", "accuracy_plot.png")
        # Save the accuracy plot as an image
        plt.savefig(accuracy_plot_path)
        plt.close()

    def get_best_model(self, x_train, y_train, x_val, y_val):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception
        """
        try:
            y_train = tf.one_hot(y_train, depth=4)
            y_val = tf.one_hot(y_val, depth=4)
            model = self.model_builder()
            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
            # Create directory to save models
            os.makedirs("models", exist_ok=True)
            # Create ModelCheckpoint object
            checkpointer = ModelCheckpoint(monitor='val_loss', mode='min',
                                           filepath=os.path.join(os.getcwd(), 'models', 'model.h5'),
                                           verbose=1, save_best_only=True)
            # Create EarlyStopping object
            early_stopping_monitor = EarlyStopping(patience=20)
            callbacks_list = [checkpointer, early_stopping_monitor]
            history = model.fit(x_train, y_train, epochs=100,
                                validation_data=(x_val, y_val),
                                callbacks=[checkpointer, early_stopping_monitor],
                                batch_size=32,
                                shuffle=True)
            self.save_loss_and_accuracy(history)
            self.logger_object.log(self.file_object,
                                   'Model training successful')
            print("Model training successful")
        except Exception as e:
            print(e)
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            raise Exception()
        return model
