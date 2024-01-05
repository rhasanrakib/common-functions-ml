
from typing import TypeVar, Generic, NamedTuple, TypedDict, Literal, Dict
import type as tp

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTEN
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import StackingClassifier


import json
import os
import pickle
import sys


class Common:
    def create_dir(self, dir: str):
        isExist = os.path.exists(dir)
        if not isExist:
            os.makedirs(dir)

    def label_encoding(self, data_frame: pd.DataFrame, features: list, column_alias="") -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        data_copy = data_frame.copy()
        label_encoders: Dict[str, LabelEncoder] = {}
        for column in features:
            le = LabelEncoder()
            data_copy[column] = le.fit_transform(data_copy[column])
            if column_alias == "":
                label_encoders[column] = le
            else:
                label_encoders[column+'_'+column_alias] = le
        return (data_copy, label_encoders)

    def default_classifier(self):
        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(),
            'XGBoost': XGBClassifier(),
            'CatBoost': CatBoostClassifier(silent=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(),
        }
        results = {}
        for name, classifier in classifiers.items():
            results[name] = classifier
        return results

    def parameter_tuning(self, classifiers: tp.ClassifierDict):
        classifiersDict = {
            'Logistic Regression': LogisticRegression(**classifiers['Logistic Regression']),
            'SVM': SVC(**classifiers['SVM']),
            'XGBoost': XGBClassifier(**classifiers['XGBoost']),
            'CatBoost': CatBoostClassifier(**classifiers['CatBoost']),
            'KNN': KNeighborsClassifier(**classifiers['KNN']),
            'Random Forest': RandomForestClassifier(**classifiers['Random Forest']),
        }
        results: Dict[Literal['Logistic Regression', 'SVM', 'XGBoost', 'CatBoost', 'KNN', 'Random Forest'],
                      LogisticRegression | SVC | XGBClassifier | KNeighborsClassifier | RandomForestClassifier] = {}
        for name, classifier in classifiersDict.items():
            results[name] = classifier
        return results


    def ensemble_stacking(self, all_models: dict, train: any, labels: any):
        result = {}
        model_names = list(all_models.keys())
        for i in range(len(model_names)):
            final_est = None
            est = []
            for name, estim in all_models.items():
                if name == model_names[i]:
                    final_est = estim
                else:
                    est.append((name, estim))
            stack = StackingClassifier(
                estimators=est, final_estimator=final_est)
            classification_res = stack.fit(train, labels)
            result[model_names[i]] = classification_res
        return result

    def fit_individual_models(self, all_models: dict, train: any, labels: any):
        result = {}
        for name, model in all_models.items():
            classification_res = model.fit(train, labels)
            result[name] = classification_res
        return result

    def upsampling(self, train, labels, sampling_strategy_: dict = None, k_neighbors=5):
        smotenc = SMOTEN(categorical_encoder=None, random_state=42,
                         sampling_strategy=sampling_strategy_, k_neighbors=k_neighbors)
        X_os_nc, y_os_nc = smotenc.fit_resample(train, labels)
        return X_os_nc, y_os_nc

    def evaluate_model(self, y_true, y_pred, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        # self.plot_confusion_matrix(y_true, y_pred, model_name)
        return {
            'model_name': model_name,
            'accuracy': '{0:.3f}'.format(accuracy),
            'precision': {
                'average_weighted': '{0:.3f}'.format(precision_score(y_true, y_pred, average='weighted')),
                'average_micro': '{0:.3f}'.format(precision_score(y_true, y_pred, average='micro')),
                'average_macro': '{0:.3f}'.format(precision_score(y_true, y_pred, average='macro')),
            },
            'recall': {
                'average_weighted': '{0:.3f}'.format(recall_score(y_true, y_pred, average='weighted')),
                'average_micro': '{0:.3f}'.format(recall_score(y_true, y_pred, average='micro')),
                'average_macro': '{0:.3f}'.format(recall_score(y_true, y_pred, average='macro')),
            },
            'f1': {
                'average_weighted': '{0:.3f}'.format(f1_score(y_true, y_pred, average='weighted')),
                'average_micro': '{0:.3f}'.format(f1_score(y_true, y_pred, average='micro')),
                'average_macro': '{0:.3f}'.format(f1_score(y_true, y_pred, average='macro')),
            },
            # 'roc':roc_auc_score(y_true, y_pred,multi_class='ovr')
        }

    def save_models(self, saved_models_folder: str, file_name: str, model):
        file_name = file_name.replace("/", "_")
        with open(f'{saved_models_folder}{file_name}.pkl', 'wb') as file:
            pickle.dump(model, file)

    def dict_to_json(self, json_output_folder: str, dictionary: dict, file_name='sample'):
        file_name = file_name.replace("/", "_")
        with open(json_output_folder+file_name+'.json', "w") as outfile:
            json.dump(dictionary, outfile, indent=4)


    def smoten_auto_up_sampling_count(self, upsampling_dict: dict):
        values = list(upsampling_dict.values())

        max_value = max(values)
        values.remove(max_value)
        second_max_value = max(values)
        difference = abs(max_value-second_max_value)
        if difference > 200:
            num_sample = difference - 50
        elif difference <= 200:
            if difference > 150 and difference <= 200:
                num_sample = 150
            elif difference > 100 and difference <= 150:
                num_sample = 100
            elif difference > 50 and difference <= 100:
                num_sample = 75
            elif difference > 25 and difference <= 50:
                num_sample = 20
            else:
                num_sample = difference-5
        re_dict = {}
        for key, value in upsampling_dict.items():
            if value == max_value:
                pass
            else:
                re_dict[key] = value+num_sample
        return upsampling_dict, re_dict

    def plot_learning_curves(self, model, X_train, X_test, y_train, y_test, path: str, file_name: str):
        train_scores, test_scores = [], []

        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_test_predict = model.predict(X_test)
            train_scores.append(roc_auc_score(y_train, y_train_predict))
            test_scores.append(roc_auc_score(y_test, y_test_predict))
        plt.plot(train_scores, "r-+", linewidth=2, label="train")
        plt.plot(test_scores, "b-", linewidth=2, label="test")
        plt.savefig(path+file_name, bbox_inches='tight')

    def calculate_min_sample(self, label):
        uniq_label = {}
        for i in label:
            if i in uniq_label:
                uniq_label[i] += 1
            else:
                uniq_label[i] = 1
        min_num_of_sample = 9999999
        for key, val in uniq_label.items():
            if val <= min_num_of_sample:
                min_num_of_sample = val
        return min_num_of_sample

