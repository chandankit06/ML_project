import os
import sys
import dill
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(xtrain,ytrain,xtest,ytest,models,param):
    try:
        # xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=42)
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs= GridSearchCV(model,para,cv=3)

            gs.fit(xtrain,ytrain)
            model.set_params(**gs.best_params_)
            model.fit(xtrain,ytrain)
             
            # model.fit(xtrain,ytrain) #Train model

            ytrainpred= model.predict(xtrain)    
            ytestpred= model.predict(xtest)    

            train_model_score= r2_score(ytrain,ytrainpred)
            test_model_score= r2_score(ytest,ytestpred)

            report[list(models.keys())[i]]=test_model_score

        return report    
            
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
