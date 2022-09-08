# models_core.py
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score

class Predict:
    # Class constructor, loads the dataset and loads the model and predict

    @staticmethod
    def predict():
        lgbm_model = pickle.load(open('best_model.pickle', 'rb'))
        test_set = pickle.load(open('test_fs_lightgbm_80.pickle', 'rb'))
        
        X_test = test_set.drop('SK_ID_CURR', axis=1)
        probability = lgbm_model.predict_proba(X_test)[:, 1] #probabilité
        prediction = lgbm_model.predict(X_test) #prediction
     
        output = pd.DataFrame({'prediction': prediction[:10],  'probability': probability[:10]})
            print(output)
            return output.to_dict(orient = 'records')
        Predict.predict()
