import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for curr_mod in models:
            reg_model = models[curr_mod]
            reg_model.fit(X_train, y_train)
            y_train_pred = reg_model.predict(X_train)
            y_test_pred = reg_model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[curr_mod] = test_model_score
            print(report)
        return report
    except Exception as err:
        CustomException(err, sys)
