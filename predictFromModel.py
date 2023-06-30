import pandas
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pickle


class prediction:
    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):
        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()
            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data = preprocessor.dropUnnecessaryColumns(data,
                                                       ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
                                                        'FTI_measured', 'TBG_measured', 'TBG', 'TSH'])
            # replacing '?' values with np.nan as discussed in the EDA part
            data = preprocessor.replaceInvalidValuesWithNull(data)
            # get encoded values for categorical data
            data = preprocessor.encodeCategoricalValuesPrediction(data)
            # Impute NULL values
            is_null_present = preprocessor.is_null_present(data)
            if is_null_present:
                data=preprocessor.impute_missing_values(data)
            # Standardize data
            continuous_features = ['age', 'T3', 'TT4', 'T4U', 'FTI']
            data = preprocessor.standardize_data(data, continuous_features)
            # Load model
            file_loader = file_methods.File_Operation(self.file_object,self.log_writer)
            model = file_loader.load_model()
            X = data.values
            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred, axis=1)
            # Load encoder to get the class names
            with open('EncoderPickle/enc.pickle', 'rb') as file:
                encoder = pickle.load(file)
            y_pred = encoder.inverse_transform(y_pred)
            data["prediction_label"] = y_pred
            path="Prediction_Output_File/Predictions.csv"
            data.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path




