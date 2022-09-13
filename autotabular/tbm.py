import lightgbm as lgb
from autotabular.eda import AutoEDA
from copy import deepcopy
from autotabular.utils import printFormattedJson
import pandas as pd

class TabularModel:
    """TabularModel class"""
    def __init__(self, max_epoch):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.label_name = None
        self.train_loop_num = 0
        self.is_multi_label = None
        self.imp_cols = None
        self.unknow_cols = None
        self.lgb_info = dict()
        self.pre_increment_preds = True
        self.first_preds = False
        self.done_training = False
        self.max_epoch = max_epoch
        self.auto_eda = AutoEDA()

    def fit(self, trainset, label_name, remaining_time_budget=None):
        
        self.train_loop_num += 1

        if self.train_loop_num == 1:
            sample_num = 500
        elif self.train_loop_num == 2:
            sample_num = 1000
        elif self.train_loop_num == 3:
            sample_num = 2000
        elif self.train_loop_num == 4:
            sample_num = 3000
        else:
            sample_num = 500*2**(self.train_loop_num-3)

        if sample_num <= trainset.shape[0]:
            self.X_train = deepcopy(trainset.drop(label_name, axis=1).loc[:sample_num-1,:])
            self.Y_train = deepcopy(trainset.loc[:sample_num-1, label_name])
            if not self.label_name:
                self.label_name = label_name
        else:
            self.pre_increment_preds = False
            if self.train_loop_num == 1:
                self.first_preds = True
            self.train_loop_num = 1

    def predict(self, x_test: pd.DataFrame, remaining_time_budget=None):
        if self.pre_increment_preds or self.first_preds:
            if self.X_test is None: self.X_test = x_test
            preds = self.simple_lgb(self.X_train, self.Y_train, self.X_test)
            if self.first_preds:
                self.first_preds = False
                self.train_loop_num = 0
        else:
            if self.train_loop_num == 1:
                self.X_test.index = -self.X_test.index - 1
                main_df = pd.concat([self.X_train, self.X_test], axis=0)
                del self.X_train, self.X_test

                eda_info = self.auto_eda.get_info(main_df)
                eda_info['is_multi_label'] = self.is_multi_label
                printFormattedJson(eda_info)
                self.data_space = TabularDataSpace(self.metadata_info, eda_info, main_df, self.Y_train, self.lgb_info)
                self.model_space = TabularModelSpace(self.metadata_info, eda_info)
                self.explore = Explore(self.metadata_info, eda_info, self.model_space, self.data_space)
            print('time', remaining_time_budget)
            self.explore.explore_space(train_loop_num=self.train_loop_num, time_remain=remaining_time_budget)
            preds = self.explore.predict()

        return preds

    def simple_lgb(self, X, y, test_x):

        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            'num_class': y.nunique(),
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": 2020,
            "num_threads": 4,
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 110,
            'subsample': 1,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            "learning_rate": 0.1
        }
        num_boost_round = 10

        print('simple lgb predict train_loop_num:', self.train_loop_num)
        if self.train_loop_num == 1:
            if X.shape[1] > 500:
                pass
                # self.sample_cols = list(set(X.columns))[::2]
                # self.unknow_cols = [col for col in X.columns if col not in self.sample_cols]
                # X = X[self.sample_cols]
                # test_x = test_x[self.sample_cols]
            if self.is_multi_label:
                pass
                # self.params['num_class'] = 2
                # all_preds = []
                # for cls in range(y.shape[1]):
                #     cls_y = y[:, cls]
                #     data = lgb.Dataset(X, cls_y)
                #     self.models[cls] = lgb.train({**self.params, **self.hyperparams}, data)
                #     preds = self.models[cls].predict(test_x)
                #     all_preds.append(preds[:, 1])
                # preds = np.stack(all_preds, axis=1)
            else:
                lgb_train = lgb.Dataset(X, y.values)
                self.model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train, num_boost_round=num_boost_round)
                preds = self.model.predict(test_x)
            self.log_feat_importances()
        else:
            num_boost_round += self.train_loop_num * 5
            num_boost_round = min(40, num_boost_round)

            if self.is_multi_label:
                pass
                # models = {}
                # all_preds = []
                # for cls in range(y.shape[1]):
                #     cls_y = y[:, cls]
                #     data = lgb.Dataset(X[self.imp_cols], cls_y)
                #     models[cls] = lgb.train({**self.params, **self.hyperparams}, data)
                #     preds = models[cls].predict(test_x[self.imp_cols])
                #     all_preds.append(preds[:, 1])
                # preds = np.stack(all_preds, axis=1)
            else:
                lgb_train = lgb.Dataset(X[self.imp_cols], y)
                model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train, num_boost_round=num_boost_round)
                preds = model.predict(test_x[self.imp_cols])
        return preds

    def log_feat_importances(self):
        if not self.is_multi_label:
            importances = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                                                                    'importances': self.model.feature_importance("gain")})
        else:
            importances = pd.DataFrame({'features': [i for i in self.models[0].feature_name()],
                                                                    'importances': self.models[0].feature_importance("gain")})

        importances.sort_values('importances', ascending=False, inplace=True)

        importances = importances[importances['importances'] > 0]
        size = int(len(importances)*0.8)
        if self.imp_cols is None:
            if self.unknow_cols is not None:
                self.imp_cols = self.unknow_cols + importances['features'].tolist()
            else:
                self.imp_cols = importances['features'].tolist()
        else:
            self.imp_cols = importances['features'].tolist()
        self.lgb_info['imp_cols'] = self.imp_cols