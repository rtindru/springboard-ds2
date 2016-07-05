import pandas as pd
import json
import os
import gc
import argparse
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

file_name_str = 'rf_fin_{}_{}_{}_{}.pkl'
out_file_name = 'rf_results.csv'
out_file_name2 = 'rf_results2.csv'

gc.enable()


class MultiPredictionModel(object):
    def __init__(self, df, xsize=0.5, ysize=0.5, xslide=0.25, yslide=0.25, xcol='x', ycol='y', th=5):
        self.preds = None
        self.df = df
        self.xsize = xsize
        self.ysize = ysize
        self.xslide = xslide
        self.yslide = yslide
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = self.df.x.max()
        self.ymax = self.df.y.max()
        self.expected = None
        self.order_expected = None
        self.actual = None
        self.result_set = {}
        self.features = ['x', 'y', 'hour', 'weekday', 'day', 'month', 'year', 'log2_accuracy']
        self.th = th
        self.windows = self.generate_windows()

    def mod_df(self, df):
        # Feature engineering
        fw = [500, 1000, 4, 3, 1. / 22., 2, 10, 10]  # feature weights (black magic here)
        df.x = df.x.values * fw[0]
        df.y = df.y.values * fw[1]
        initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                                   for mn in df.time.values)
        df['hour'] = d_times.hour * fw[2]
        df['weekday'] = d_times.weekday * fw[3]
        df['day'] = (d_times.dayofyear * fw[4]).astype(int)
        df['month'] = d_times.month * fw[5]
        df['year'] = (d_times.year - 2013) * fw[6]
        df['log2accu'] = np.log2(df.accuracy) * fw[7]

        return df

    def frange(self, x, y, jump):
        while x < y:
            yield x
            x += jump
        yield y

    def generate_windows(self):
        ranges = []
        result = []

        xmin, xmax = round(self.df.x.min(), 3), round(self.df.x.max(), 3)
        ymin, ymax = round(self.df.y.min(), 3), round(self.df.y.max(), 3)
        xranges = list([round(x, 3) for x in self.frange(xmin, xmax - self.xsize, self.xslide)])
        yranges = list([round(y, 3) for y in self.frange(ymin, ymax - self.ysize, self.yslide)])
        ylen = len(yranges)
        for x in xranges:
            subrange = [x] * ylen
            ranges.extend(zip(subrange, yranges))

        for x1, y1 in ranges:
            x2, y2 = round(x1 + self.xsize, 3), round(y1 + self.ysize, 3)
            result.append(((x1, y1), (x2, y2)))

        return result

    def find_x_window(self, x):
        xs = max(0, x - self.xsize)
        x0 = 0

        while x0 < xs:
            x0 += self.xslide
        if x0 >= self.xmax - self.xsize:
            x0 = self.xmax - self.xsize
        return round(x0, 3)

    def find_y_window(self, y):
        ys = max(0, y - self.ysize)
        y0 = 0

        while y0 < ys:
            y0 += self.yslide
        if y0 >= self.ymax - self.ysize:
            y0 = self.ymax - self.ysize
        return round(y0, 3)

    def process_cell(self, df_cell_train, df_cell_test, window):
        row_ids = df_cell_test.index
        le = LabelEncoder()
        y = le.fit_transform(df_cell_train.place_id.values)
        X = df_cell_train.loc[:, self.features].values.astype(int)
        X_test = df_cell_test[:, self.features].values.astype(int)

        clf = KNeighborsClassifier(n_neighbors=25, weights='distance',
                                   metric='manhattan')
        clf.fit(X, y)
        y_pred = clf.predict_proba(X_test)
        pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:, ::-1][:, :3])
        return pred_labels, row_ids

    def process_cells(self, df_tran, df_test):
        self.preds = np.zeros((len(df_test), 3), dtype=int)
        for i, window in enumerate(self.windows):
            print 'Processing Window: {} of {}'.format(i + 1, len(self.windows))
            (x1, y1), (x2, y2) = window
            train_df = self.mod_df(df_tran[(df_tran[self.xcol] >= x1) & (df_tran[self.xcol] <= x2) &
                                           (df_tran[self.ycol] >= y1) & (df_tran[self.ycol] <= y2)])
            test_df = self.mod_df(df_test[(df_test[self.xcol] >= x1) & (df_test[self.xcol] <= x2) &
                                          (df_test[self.ycol] >= y1) & (df_test[self.ycol] <= y2)])

            pred_labels, row_ids = self.process_cell(train_df, test_df, window)
            self.preds[row_ids] = pred_labels

        df_aux = pd.DataFrame(self.preds, dtype=str, columns=['l1', 'l2', 'l3'])
        # Concatenating the 3 predictions for each sample
        ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

        # Writting to csv
        print 'Writing to CSV'
        ds_sub.name = 'place_id'
        ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')

    def round_off(self, x):
        return round(x, 3)


def run(xsize, ysize, xstep, ystep, th):
    print 'Loading DataFrame'
    df_train = pd.read_csv('../Kaggle_Datasets/Facebook/train.csv')
    df_test = pd.read_csv('../Kaggle_Datasets/Facebook/test.csv')

    place_counts = df_train.place_id.value_counts()
    mask = (place_counts[df_train.place_id.values] >= th).values
    df_train = df_train.loc[mask]

    print 'Initializing PredictionModel class'
    pred_model = MultiPredictionModel(df_train, xsize, ysize, xstep, ystep, 'x', 'y', )
    print 'Init done'
    print pred_model.windows

    print 'Training Model'
    pred_model.process_cells(df_train, df_test)
    print 'Done Training'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xsize", help="Size of X", type=float)
    parser.add_argument("ysize", help="Size of Y", type=float)
    parser.add_argument("xstep", help="Step of X", type=float)
    parser.add_argument("ystep", help="Step of Y", type=float)
    parser.add_argument("th", help="Threshold", type=int)
    args = parser.parse_args()
    xsize = round(args.xsize, 3)
    ysize = round(args.ysize, 3)
    xslide = round(args.xstep, 3)
    yslide = round(args.ystep, 3)
    th = round(args.tg, 3)
    print run(xsize, ysize, xslide, yslide, th)
