# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()
            """Data preprocessing"""
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            # removing unwanted columns
            data = preprocessor.dropUnnecessaryColumns(data,
                                                       ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
                                                        'FTI_measured', 'TBG_measured', 'TBG', 'TSH'])
            # replacing '?' values with np.nan
            data = preprocessor.replaceInvalidValuesWithNull(data)
            # get encoded values for categorical data
            data = preprocessor.encodeCategoricalValues(data)
            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='Class')
            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(X)
            # if missing values are there, replace them appropriately.
            if is_null_present:
                X = preprocessor.impute_missing_values(X)  # missing value imputation
            # Handle imbalanced data
            X, Y = preprocessor.handleImbalanceDataset(X, Y)
            # Standardize data
            continuous_features = ['age', 'T3', 'TT4', 'T4U', 'FTI']
            X = preprocessor.standardize_data(X, continuous_features)
            # Separate train, val and test set
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                                stratify=Y,
                                                                random_state=32)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                               test_size=0.2,
                                                               stratify=y_train,
                                                               random_state=32)
            # Set the print options to display all columns
            x_train, y_train = x_train.values, y_train.values
            x_val, y_val = x_val.values, y_val.values
            x_test, y_test = x_test.values, y_test.values

            # Find the best model
            model_finder = tuner.Model_Finder(self.file_object, self.log_writer)
            model = model_finder.get_best_model(x_train, y_train, x_val, y_val)

            # Evaluate model
            model_finder.evaluate(model, x_test, y_test)

            # saving the best model to the directory.
            '''file_operator = file_methods.File_Operation(self.file_object,
                                                        self.log_writer)
            file_operator.save_model(model)'''

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            print(e)
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
