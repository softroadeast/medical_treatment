#coding=gbk
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_spilt=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

class MedicalTreament:
    def __init__(self):
        self.train_file = "d_train_20180102.csv"
        self.test_file = "d_test_A_20180102.csv"
        self.model_name = "xgboost"
        self.feature_series = None
        self.find_best_param = 0
        self.fillna_with_total_data = 0
        self.preprocess_data = 0
        self.add_poly_feature = 1
        self.replace_outlier = 0
        self.use_ensemble = 0
        self.k_folder = 5
        self.ensemble_clf = [ExtraTreesRegressor(random_state=2017),RandomForestRegressor(random_state=2017),\
                             GradientBoostingRegressor(random_state=2017),XGBRegressor(random_state=2017)]

    def read_file(self):
        self.train_df = pd.read_csv(self.train_file,encoding="GBK")
        self.test_df = pd.read_csv(self.test_file, encoding="GBK")

    def work_with_na(self):
        self.train_y = self.train_df.pop('血糖').values
        if self.fillna_with_total_data:
            total_df = pd.concat([self.train_df, self.test_df])
            total_gender_df = total_df['性别']
            total_grouped = total_df.groupby('性别')
            f = lambda x: x.fillna(x.median())
            total_grouped_df = total_grouped.transform(f)
            self.train_df = total_grouped_df.iloc[:self.train_df.shape[0], :]
            self.test_df = total_grouped_df.iloc[self.train_df.shape[0]:, :]
            train_gender_df = total_gender_df.iloc[:self.train_df.shape[0]]
            test_gender_df = total_gender_df.iloc[self.train_df.shape[0]:]
            self.train_df = pd.concat([train_gender_df, self.train_df], axis=1)
            self.test_df = pd.concat([test_gender_df, self.test_df], axis=1)
        else:
            self.train_df = self.train_df.fillna(self.train_df.median())
            self.test_df = self.test_df.fillna(self.train_df.median())

    def work_with_gender(self):
        self.train_df.loc[:,'性别'].replace({"男":1,"女":0,"??":0},inplace=True)
        self.test_df.loc[:,'性别'].replace({"男": 1, "女": 0, "??": 0}, inplace=True)

    def work_with_date(self):
        self.train_df = self.train_df.loc[:,self.train_df.dtypes!=object]
        self.test_df = self.test_df.loc[:, self.test_df.dtypes != object]

    def work_with_id(self):
        self.train_df.pop('id')
        self.test_df.pop('id')

    def prepare_data(self):
        if self.replace_outlier:
            self.train_part_df = self.train_df.iloc[:,2:]
            self.test_part_df = self.test_df.iloc[:,2:]
            self.train_part_df = self.train_part_df.clip(self.train_part_df.mean()-3*self.train_part_df.std(),\
                                                         self.train_part_df.mean()+3*self.train_part_df.std(),\
                                                         axis=1)
            self.test_part_df = self.test_part_df.clip(self.test_part_df.mean() - 3 * self.test_part_df.std(), \
                                                         self.test_part_df.mean() + 3 * self.test_part_df.std(), \
                                                         axis=1)
            self.train_df = pd.concat([self.train_df.iloc[:,:2],self.train_part_df],axis=1)
            self.test_df = pd.concat([self.test_df.iloc[:,:2],self.test_part_df],axis=1)
        self.train_X = self.train_df.values
        self.test_X = self.test_df.values

    def select_feature_by_mutual_info(self):
        mutual_info_np = mutual_info_regression(self.train_X,self.train_y)
        mutual_info_series = pd.Series(mutual_info_np, index=self.train_df.columns)
        self.feature_series = mutual_info_series

    def select_feature_by_corrcoef(self):
        corr_values = []
        for i in range(self.train_X.shape[1]):
            corr_values.append(abs(np.corrcoef(self.train_X[:,i], self.train_y)[0, 1]))
        corr_series = pd.Series(corr_values, index=self.train_df.columns)
        self.feature_series = corr_series

    def select_feature_by_ET(self):
        clf = ExtraTreesRegressor()
        clf.fit(self.train_X, self.train_y)
        feature_importance = clf.feature_importances_
        et_series = pd.Series(feature_importance, index=self.train_df.columns)
        self.feature_series = et_series
        print(self.feature_series)

    def prepare_data_after_select_feature(self):
        self.train_df = self.train_df.loc[:,self.feature_series > 0.01]
        self.train_X = self.train_df.values
        self.test_df = self.test_df[self.train_df.columns]
        self.test_X = self.test_df.values

        if self.preprocess_data:
            scaler = StandardScaler()
            self.train_X = scaler.fit_transform(self.train_X)
            self.test_X = scaler.transform(self.test_X)

        if self.add_poly_feature:
            poly = PolynomialFeatures(2)
            print(self.train_X.shape)
            self.train_X = poly.fit_transform(self.train_X)
            print(self.train_X.shape)
            self.test_X = poly.transform(self.test_X)
            """
            train_X = self.train_X[:, 0].reshape(self.train_X.shape[0],1)
            test_X = self.test_X[:, 1].reshape(self.test_X.shape[0], 1)
            for i in range(self.train_X.shape[1]):
                if abs(np.corrcoef(self.train_X[:, i], self.train_y)[0, 1])>0.1:
                    train_X = np.hstack((train_X,self.train_X[:, i].reshape(self.train_X.shape[0],1)))
                    test_X = np.hstack((test_X, self.test_X[:, i].reshape(self.test_X.shape[0], 1)))
            self.train_X = train_X[:,1:]
            self.test_X = test_X[:,1:]
            """


    def train_and_predict(self):
        if self.use_ensemble:
            stack_train = np.zeros((self.train_X.shape[0], len(self.ensemble_clf)))
            stack_test = np.zeros((self.test_X.shape[0], len(self.ensemble_clf)))

            kf = KFold(n_splits=self.k_folder,random_state=2017)
            for i,clf in enumerate(self.ensemble_clf):
                train_test_y = np.zeros((self.train_X.shape[0]))
                test_test_y = np.zeros((self.test_X.shape[0], self.k_folder))
                for j,(train_index,test_index) in enumerate(kf.split(self.train_X)):
                    train_train_X = self.train_X[train_index]
                    train_test_X = self.train_X[test_index]
                    train_train_y = self.train_y[train_index]

                    clf.fit(train_train_X,train_train_y)
                    train_test_y[test_index] = clf.predict(train_test_X)
                    test_test_y[:,j] = clf.predict(self.test_X)
                stack_train[:,i] = train_test_y
                stack_test[:,i] = test_test_y.mean(axis=1)
            stack_clf = XGBRegressor()
            stack_clf.fit(stack_train,self.train_y)
            self.test_y = stack_clf.predict(stack_test)
            return

        if self.model_name == "LR":
            clf = LinearRegression()
        elif self.model_name == "xgboost":
            clf = XGBRegressor()
            if self.find_best_param:
                test_params = {
                    'max_depth': [2,3, 4, 5, 6],
                    'learning_rate': [0.25,0.5,0.1, 0.3],
                    'n_estimators': [10,25,50, 100, 200]
                }
                mean_squared_error_scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)
                gs = GridSearchCV(estimator=clf, param_grid=test_params, cv=4, verbose=2,
                                  scoring=mean_squared_error_scorer, n_jobs=2)
                gs.fit(self.train_X, self.train_y)
                print('======xgboost==== Best Results ================')
                print('best params: {}, best score: {}'.format(gs.best_params_, gs.best_score_))
                print('=============== End ================')
                best_params = gs.best_params_
                with open("medical_treament_find_best_param.txt",'a+') as fh:
                    fh.write("{} {} best params: {} best score: {}\n".format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), self.model_name, gs.best_params_,
                    gs.best_score_))

            else:
                best_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
            clf = XGBRegressor(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'],
                               max_depth=best_params['max_depth'])
        elif self.model_name == "ET":
            clf = ExtraTreesRegressor()
        elif self.model_name == "network":
            clf = MLPRegressor(solver='lbfgs')
            if self.find_best_param:
                test_params = {
                    'hidden_layer_sizes': [20,50, 100, 150, 200],
                    'activation': ['identity','logistic','tanh', 'relu'],
                    'learning_rate_init':[0.0001,0.005,0.001,0.002],
                    'early_stopping': [True,False]
                }
                mean_squared_error_scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)
                gs = GridSearchCV(estimator=clf, param_grid=test_params, cv=4, verbose=2,
                                  scoring=mean_squared_error_scorer, n_jobs=2)
                gs.fit(self.train_X, self.train_y)
                print('======network==== Best Results ================')
                print('best params: {}, best score: {}'.format(gs.best_params_, gs.best_score_))
                print('=============== End ================')
                best_params = gs.best_params_
                with open("medical_treament_find_best_param.txt",'a+') as fh:
                    fh.write("{} {} best params: {} best score: {}\n".format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), self.model_name, gs.best_params_,
                    gs.best_score_))

            else:
                best_params = {'activation': 'identity', 'early_stopping': True, 'hidden_layer_sizes': 200,'learning_rate_init': 0.005}
            clf = MLPRegressor(hidden_layer_sizes=best_params['hidden_layer_sizes'],solver='lbfgs',\
                               activation=best_params['activation'],learning_rate_init=best_params['learning_rate_init'],\
                               early_stopping=best_params['early_stopping'])
        else:
            print("no this model!")
            return
        clf.fit(self.train_X, self.train_y)
        self.test_y = clf.predict(self.test_X)
        print("train successfully!")


    def genarate_result_file(self):
        test_3_y = []
        for i in range(len(self.test_y)):
            test_3_y.append("%.3f" % self.test_y[i])
            #print(test_3_y[i])

        result_df = pd.Series(test_3_y)
        result_df.to_csv("result_{}_{}.csv".format(self.model_name,time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())),header=None,index=False)
        print("genarate file successfully!")


    def start(self):
        self.read_file()
        self.work_with_na()
        self.work_with_gender()
        self.work_with_date()
        self.work_with_id()
        self.prepare_data()
        self.select_feature_by_corrcoef()
        self.prepare_data_after_select_feature()
        self.train_and_predict()
        self.genarate_result_file()


if __name__ == "__main__":
    medical_treament = MedicalTreament()
    medical_treament.start()