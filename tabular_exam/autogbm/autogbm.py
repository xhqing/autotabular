#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

from autogbm.auto_ingestion import data_io
from autogbm.auto_ingestion.dataset import AutoDLDataset
from autogbm.convertor.tabular_to_tfrecords import autotabular_2_autodl_format
from autogbm.auto_models.auto_tabular.model import Model as TabularModel
from autogbm.auto_ingestion.pure_model_run import run_single_model

def autogbm_run(df: pd.DataFrame, label: pd.Series, formatted_dir: str="./formatted_data"):

    autotabular_2_autodl_format(formatted_dir=formatted_dir, data=df, label=label)

    new_dataset_dir = formatted_dir + "/" + os.path.basename(formatted_dir)
    datanames = data_io.inventory_data(new_dataset_dir)

    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))
    
    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))

    max_epoch = 50
    time_budget = 1200
    model = TabularModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)
