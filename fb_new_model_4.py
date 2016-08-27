import math
import pandas as pd
import json
import os
import gc
import argparse
import numpy as np
import csv

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

xybins_file_name_str = 'xy_model_{}.pkl'
out_file_name = 'rf_results.csv'
out_file_name2 = 'rf_results2.csv'

x_bins = 200
y_bins = 50
x_std = 0.01
y_std = 0.005
max_time = 1006589.0
time_bins = 4
time_period = 24 * 60.0
grid_size = 10.0

gc.enable()


class MultiPredictionModel(object):
    def __init__(self, df):
        self.df = df
        self.expected = None
        self.order_expected = None
        self.actual = None
        self.result_set = {}
        self.xy_bins = []

        self.features = ['x', 'y', 'hour', 'day', 'week', 'month', 'year', 'log2_accuracy',
                         'log10_accuracy', 'day_r', 'week_r', 'month_r', 'year_r',
                         'time_bin', 'dist_xy_center', 'dist_x_center', 'dist_y_center', ]

        self.models = [
            ('rf1', RandomForestClassifier(n_jobs=-1, n_estimators=15, min_samples_split=4)),
            ('gb1', GradientBoostingClassifier(min_samples_split=4, min_samples_leaf=2)),
            ('gnb', GaussianNB()),
        ]

        self.df = self.feature_engineering(self.df)
        self.xy_bins = self.df.xy_bin.unique()

    def feature_engineering(self, df):
        df.loc[:, 'hours'] = df.time / float(60)
        df.loc[:, 'hour'] = df.hours % 24 + 1
        df.loc[:, 'hour_r'] = np.floor(df.hour)

        df.loc[:, 'days'] = df.time / float(60 * 24)
        df.loc[:, 'day'] = df.days % 7 + 1
        df.loc[:, 'day_r'] = np.floor(df.day)

        df.loc[:, 'weeks'] = df.time / float(60 * 24 * 7)
        df.loc[:, 'week'] = df.weeks % 52 + 1
        df.loc[:, 'week_r'] = np.floor(df.week)

        df.loc[:, 'months'] = df.time / float(60 * 24 * 30)
        df.loc[:, 'month'] = df.months % 12 + 1
        df.loc[:, 'month_r'] = np.floor(df.month)

        df.loc[:, 'year'] = df.time / float(60 * 24 * 365) + 1
        df.loc[:, 'year_r'] = np.floor(df.year)

        df.loc[:, 'log2_accuracy'] = np.log2(df.accuracy) * 10
        df.loc[:, 'log10_accuracy'] = np.log10(df.accuracy) * 10

        df.loc[:, 'x_bin'] = df.x.apply(self.apply_x_bin)
        df.loc[:, 'y_bin'] = df.y.apply(self.apply_y_bin)
        df.loc[:, 'time_bin'] = df.time.apply(self.apply_time_bin)

        df.loc[:, 'xy_bin'] = df.x_bin * y_bins + df.y_bin
        df.loc[:, 'xytime_bin'] = df.xy_bin * time_bins + df.time_bin

        df.loc[:, 'x_center'] = df.x.apply(self.apply_x_center)
        df.loc[:, 'y_center'] = df.y.apply(self.apply_y_center)

        df.loc[:, 'dist_x_center'] = df.x.apply(self.apply_dist_x_center)
        df.loc[:, 'dist_y_center'] = df.y.apply(self.apply_dist_y_center)
        df.loc[:, 'dist_xy_center'] = np.sqrt((df.dist_x_center * df.dist_x_center) + (df.dist_y_center * df.dist_y_center))
        df.loc[:, 'dist_time'] = df.time.apply(self.apply_dist_time)

        return df

    def apply_x_bin(self, x):
        return int(math.floor((x * x_bins) / grid_size))

    def apply_y_bin(self, y):
        return int(math.floor((y * y_bins) / grid_size))

    def apply_time_bin(self, time):
        return int(math.floor((time % time_period) / (time_period / time_bins)))

    def apply_x_center(self, x):
        x_bin = self.apply_x_bin(x)
        x_center = (x_bin + 0.5) * (grid_size / x_bins)
        return x_center

    def apply_y_center(self, y):
        y_bin = self.apply_y_bin(y)
        y_center = (y_bin + 0.5) * (grid_size / y_bins)
        return y_center

    def apply_dist_time(self, time):
        return math.exp(4.0 * time / max_time)

    def apply_dist_x_center(self, x):
        x_center = self.apply_x_center(x)
        dist = (x - x_center) / x_std
        return math.exp(-(dist * dist))

    def apply_dist_y_center(self, y):
        y_center = self.apply_y_center(y)
        dist = (y - y_center) / y_std
        return math.exp(-(dist * dist))

    def train(self):
        for bin_id in sorted(self.xy_bins):
            file_name = xybins_file_name_str.format(bin_id)
            print 'Training model: {} of {}'.format(bin_id, max(self.xy_bins))
            df = self.df
            wdf = df[df.xy_bin == bin_id]
            X = wdf[self.features]
            y = wdf.place_id

            model = VotingClassifier(self.models)
            model.fit(X, y)
            joblib.dump(model, file_name, compress=3, )

    def load_xy_model(self, xy_bin):
        try:
            file_name = xybins_file_name_str.format(xy_bin)
            model = joblib.load(file_name)
            return model
        except:
            return None

    def predict(self, df, test=False):
        df = df.sort_values('row_id')

        if test:
            self.expected = dict(zip(df.row_id, df.place_id))
            self.order_expected = df.place_id

        df = self.feature_engineering(df)

        xy_bins = sorted(df.xy_bin.unique())
        for bin_id in xy_bins:
            print 'Predicting Model: {} of {}'.format(bin_id, max(xy_bins))
            model = self.load_xy_model(bin_id)
            wdf = df[df.xy_bin == bin_id]

            if len(wdf) == 0:
                continue
            if model is None:
                for i in xrange(len(wdf)):
                    row_id = wdf.row_id.iloc[i]
                    self.result_set[row_id] = [0, 0, 0]
                continue
            
            X = wdf[self.features]
            predictions = model.predict_proba(X)

            for i in xrange(len(wdf)):
                row_id = wdf.row_id.iloc[i]
                indices = np.argsort(predictions[i])[-3:][::-1]
                places = model.classes_[indices]
                self.result_set[row_id] = places
                """ 
                try:
                    places = model.classes_[indices]
                    self.result_set[row_id] = places
                except:
                    place = model.predict(X.iloc[i])
                    self.result_set[row_id] = [place] * 3
                """
 
            model = None
            del model
            gc.collect()

        if test:
            self.actual = (self.result_set[x][0] for x in sorted(self.result_set.keys()))

        return self.result_set

    def write_result(self, ):
        with open(out_file_name, 'w') as outfile:
            csv_writer = csv.writer(outfile, delimiter=',')
            for row_id, place_list in self.result_set.iteritems():
                place_str = ' '.join((str(x) for x in place_list))
                csv_writer.writerow([row_id, place_str])

    def score(self):
        correct_count = 0
        mean_score = 0
        total_count = len(self.expected)
        for row_id, place_id in self.expected.iteritems():
            actual = self.result_set[row_id]
            if place_id == actual[0]:
                correct_count += 1
            if actual[0] == place_id:
                mean_score += 1
            elif len(actual) >= 2 and actual[1] == place_id:
                mean_score += 0.5
            elif len(actual) >= 3 and actual[2] == place_id:
                mean_score += 0.3
            else:
                mean_score += 0

        o = pd.Series(self.order_expected)
        n = pd.Series(self.actual)
        print 'Simple Score Multi Pred: {}'.format((sum(o == n) / float(len(o))) * 100)
        print 'Simple Score: {}'.format((correct_count / float(total_count)) * 100)
        print 'Mean Score: {}'.format((mean_score / float(total_count)) * 100)
        return mean_score


def run():
    try:
        os.remove(out_file_name)
    except:
        pass

    print 'Loading DataFrame'
    df_train = pd.read_csv('../Kaggle_Datasets/Facebook/train_0_0.25.csv')
    df_test = pd.read_csv('../Kaggle_Datasets/Facebook/test_0_0.25.csv')

    # df_train = df_train.loc[(df_train.x <= 0.4) & (df_train.y <= 0.2), :]
    # df_test = df_test.loc[(df_test.x <= 0.4) & (df_test.y <= 0.2), :]

    # print 'Splitting train and test data'
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=11)
    # train, cv = train_test_split(train, test_size=0.25, random_state=2)

    # df_train = None; del df_train; gc.collect()

    print 'Initializing PredictionModel class'
    pred_model = MultiPredictionModel(df_train)
    print 'Init done'

    print pred_model.df.head()

    print 'Training Model'
    pred_model.train()
    print 'Done Training'

    for i in xrange(learn_times):
        df_test = df_test.sort_values('row_id')
        df_test_preds = df_test.copy()
        test_preds = pred_model.predict_single(df_test, test=True)
        res = []
        for key in sorted(test_preds.keys()):
            res.append(test_preds[key])
        df_test_preds['place_id'] = res
        pred_model.partial_fit(df_test_preds)

    print 'Predicting on test data'
    pred_model.predict(df_test, test=True)
    print 'Done predicting'


    print 'Scoring Data'
    pred_model.score()
    print 'Done Scoring'

    # print 'Predicting on read data'
    # print pred_model.predict(df_test)
    # print 'Done predicting'
    #
    print 'Print Writing Results'
    pred_model.write_result()
    print 'Done writing results'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print run()
