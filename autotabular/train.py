#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py: auto_train func, the entry of autotabular
Created by xuhuaqing.
"""

import pandas as pd
import os
import time
from sklearn.preprocessing import OneHotEncoder
from autotabular.scores import nauc, acc
from autotabular.exception import *
from autotabular.tbm import TabularModel
from copy import deepcopy
import shutil

def pandas_dtype_check(dataset: pd.DataFrame):
    bad_fields = None
    PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float'}
    data_dtypes = dataset.dtypes
    if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
        bad_fields = [dataset.columns[i] for i, dtype in
                      enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]
    return bad_fields

def auto_train(train_set: pd.DataFrame, test_set: pd.DataFrame, label_name: str, task: str="classification", metric: str="nauc", 
                autotabular_info_path: str="./autotabular_info", gpu: bool=False, time_budget: int=1200, max_epoch: int=50):
    """
    param `train_set` and `test_set`: 
        1. should be pandas dataframe;
        2. it's columns including `label_name`;
        3. all columns' name should be string;
        4. `train_set.dtypes` and `test_set.dtypes` must be int or float
        5. the index must be start from 0,1,2,3,...
    
    param `label_name`:
        1. the type of `label_name` must be string;
        2. it must be in the `train_set`'s and `test_set`'s columns

    param `task`:
        1. the type of `task` must be string;
        2. it is in ["classification", "regression"];
        
    param `metric`:
        1. the type of `metric` must be string;
        2. the defult value is "nauc";
        3. if `task` is "classification", `metric` in ["nauc", "acc", "auc", "recall", "precission", "f1-score"],
           if `task` is "regression" `metric` in ["mse", "mae"]

    param `autotabular_info_path`:
        1. the type of `autotabular_info_path` must be string;
        2. defult value is "./autotabular_info".

    param `gpu`:
        1. the type of `gpu` must be bool;
        2. the defult value of `gpu` is False;

    param `time_budget`:
        the uplimit training time, the default value is 1200 seconds(20min). 

    param `max_epoch`:
        the max number of training round, default is 50 rounds.
    """

    if not isinstance(train_set, pd.DataFrame):
        raise TypeError("\033[01;31;01mThe data type of `train_set` must be pandas dataframe!\033[01;00;00m ")
    
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)

    if label_name not in train_set or label_name not in test_set:
        raise ValueError("\033[01;31;01mThe `label_name` must be in the columns of `train_set` and `test_set`!\033[01;00;00m ")

    for col in train_set:
        if not isinstance(col, str):
            raise TypeError(f"\033[01;31;01mAll columns name of `train_set` must be string, while {col} is not string type!\033[01;00;00m ")

    bad_fields = pandas_dtype_check(train_set)
    if bad_fields:
        msg = """\033[01;31;01m`train_set.dtypes` for data must be int or float.
                    Did not expect the data types in fields: """
        raise ValueError(msg + ', '.join(bad_fields) + "\033[01;31;01m")

    for col in test_set:
        if not isinstance(col, str):
            raise TypeError(f"\033[01;31;01mAll columns name of `test_set` must be string, while {col} is not string type!\033[01;00;00m ")

    bad_fields = pandas_dtype_check(test_set)
    if bad_fields:
        msg = """\033[01;31;01m`test_set.dtypes` for data must be int or float.
                    Did not expect the data types in fields: """
        raise ValueError(msg + ', '.join(bad_fields) + "\033[01;31;01m")
                    
    if not isinstance(task, str):
        raise TypeError(f"\033[01;31;01mThe type of `task` must be string, while {task} is not string type!\033[01;31;01m")
    if task not in ["classification", "regression"]:
        raise ValueError(f"\033[01;31;01m`task` must be \"multi\" or \"regression\", while `task` is \"{task}\"!\033[01;31;01m")

    if not isinstance(metric, str):
        raise TypeError(f"\033[01;31;01mThe type of `metric` must be string, while {metric} is not string type!\033[01;31;01m")

    if task == "classification":
        support_metrics = ["nauc", "acc", "auc", "recall", "precission", "f1-score"]
        if metric not in support_metrics:
            msg = """\033[01;31;01mCurrently, for classification task, only support metrics are: """
            msg += ", ".join(support_metrics)
            raise ValueError(msg+", "+f"while you input `metric` is \"{metric}!\"\033[01;31;01m")
    if task == "regression":
        support_metrics = ["mse", "mae"]
        if metric not in support_metrics:
            msg = """\033[01;31;01mCurrently, for regression task, only support metrics are: """
            msg += ", ".join(support_metrics)
            raise ValueError(msg+", "+f"while you input `metric` is {metric}!\033[01;31;01m")
    
    if not isinstance(autotabular_info_path, str):
        raise TypeError(f"""\033[01;31;01mThe type of `autotabular_info_path` must be string, 
                        while {autotabular_info_path} is not string type!\033[01;31;01m""")
    try:
        os.makedirs(autotabular_info_path)
    except FileExistsError:
        shutil.rmtree(autotabular_info_path)
        os.makedirs(autotabular_info_path)
        
    if not isinstance(gpu, bool):
        raise TypeError("""\033[01;31;01m`gpu` value must be bool type!\033[01;31;01m""")
    if gpu:
        raise NotsupportError("""\033[01;31;01mSorry, not support gpu yet!\033[01;31;01m""")

    if task == "regression":
        raise NotsupportError("""\033[01;31;01mSorry, not support regression task yet!\033[01;31;01m""")
    elif task == "classification":
        label_nunique = train_set[label_name].nunique()
        if label_nunique < 2:
            raise LabelError(f"""\033[01;31;01m`label_nunique` should be at least 2, but label have only {label_nunique} kind of value!\033[01;31;01m""")
        if label_nunique != test_set[label_name].nunique():
            raise LabelError(f"""\033[01;31;01mtrain_set label nunique not the same as test_set label nunique!\033[01;31;01m""")
    
        model = TabularModel(max_epoch=max_epoch)
        start_time = int(time.time())
        trainset = deepcopy(train_set)
        x_test = deepcopy(test_set.drop(label_name, axis=1))
        for i in range(max_epoch):
            remaining_time_budget = start_time + time_budget - int(time.time())
            # import pdb
            # pdb.set_trace()
            model.fit(trainset=trainset, label_name=label_name, remaining_time_budget=remaining_time_budget)
            remaining_time_budget = start_time + time_budget - int(time.time())
            y_pred = model.predict(x_test=x_test, remaining_time_budget=remaining_time_budget)

            ohe_y_test = OneHotEncoder(categories='auto').fit_transform([[ele] for ele in test_set[label_name]]).toarray()
            nauc_score = nauc(y_test=ohe_y_test, prediction=y_pred)
            acc_score = acc(y_test=ohe_y_test, prediction=y_pred)
            print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))

            if i == 0: break



