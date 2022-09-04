# APP FastAPI
# API renvoie le score d’un client (moteur d’inférence)

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, \
    fbeta_score, make_scorer
from pydantic import BaseModel
from models_core_Marwa import *


# 2. Create the app and model objects
api = FastAPI(
    title="LGBM Prediction",
    description="Predict score",
    version="1.0.0")

model = Predict()


@api.get("/Marwa")
async def current_user():
    result = {
            'status' : 'success',
            'message' : (f"Hello ! Welcome to Fast API.")
        }
    return result

# 3. Prediction

@api.post('/predict')
async def make_predictions():
    """
    Effectuer des prédictions sur de nouvelles données en provenance d'un fichier pickle
    """
    
    return model.predict()






