from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction

# Create a FastAPI app
app = FastAPI()


# Class to store path of the folder
class FolderPath(BaseModel):
    folderPath: str


# Route for prediction
@app.post("/predict")
async def predict_route_client(folder_path: FolderPath):
    try:
        path = 'Prediction_Batch_files'
        if folder_path.folderPath != "":
            path = folder_path.folderPath
        pred_val = pred_validation(path)  # object initialization
        pred_val.prediction_validation()  # calling the prediction_validation function
        pred = prediction(path)  # object initialization
        # predicting for dataset present in database
        path = pred.predictionFromModel()
        return {"message": f"Prediction File created at {path}!!!"}
    except ValueError as ve:
        return {"error": f"Error Occurred! {ve}"}
    except KeyError as ke:
        return {"error": f"Error Occurred! {ke}"}
    except Exception as e:
        return {"error": f"Error Occurred! {e}"}


# Route for training
@app.post("/train")
async def train_route_client(folder_path: FolderPath):
    try:
        path = "Training_Batch_Files"
        if folder_path.folderPath != "":
            path = folder_path.folderPath
        train_val_obj = train_validation(path)  # object initialization
        train_val_obj.train_validation()  # calling the training_validation function
        train_model_obj = trainModel()  # object initialization
        train_model_obj.trainingModel()  # training the model for the files in the table
        return {"message": "Training successful!!"}
    except ValueError as ve:
        return {"error": f"Error Occurred! {ve}"}
    except KeyError as ke:
        return {"error": f"Error Occurred! {ke}"}
    except Exception as e:
        return {"error": f"Error Occurred! {e}"}
