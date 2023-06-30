import os
import tensorflow as tf


class File_Operation:
    """
    This class shall be used to save the model after training
    and load the saved model for prediction.
    """
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def save_model(self, model):
        """
        Method Name: save_model
        Description: Save the model file to directory
        Outcome: File gets saved
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the save_model method of the File_Operation class')
        try:
            model.save(os.path.join("models", "model.h5"))
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            raise Exception()

    def load_model(self):
        """
        Method Name: load_model
        Description: load the model file to memory
        Output: The Model file loaded in memory
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the load_model method of the File_Operation class')
        try:
            model = tf.keras.models.load_model(os.path.join("models", "model.h5"))
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            raise Exception()
        return model