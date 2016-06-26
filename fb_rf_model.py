import pandas as pd
import json
import os
import gc
import argparse
import numpy as np
import csv

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

file_name_str = 'rf_fin_{}_{}_{}_{}.pkl'
out_file_name = 'rf_results.csv'
out_file_name2 = 'rf_results2.csv'

gc.enable()


class MultiPredictionModel(object):

    def __init__(self, df, xsize=0.5, ysize=0.5, xslide=0.25, yslide=0.25, xcol='x', ycol='y', n_estimators=10, n_jobs=1):
        self.df = df
        self.xsize = xsize
        self.ysize = ysize
        self.xslide = xslide
        self.yslide = yslide
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = self.df.x.max()
        self.ymax = self.df.y.max()
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.expected = None
        self.order_expected = None
        self.actual = None
        self.result_set = {}
        self.features = ['x', 'y', 'accuracy', 'hour', 'day', 'week', 'month', 'year']

        self.mod_df(self.df)
        self.windows = self.generate_windows()

    def mod_df(self, df):
        df.loc[:, 'hours'] = df.time / float(60)
        df.loc[:, 'hour'] = df.hours % 24 + 1

        df.loc[:, 'days'] = df.time / float(60*24)
        df.loc[:, 'day'] = df.days % 7 + 1

        df.loc[:, 'weeks'] = df.time / float(60*24*7)
        df.loc[:, 'week'] = df.weeks % 52 + 1

        df.loc[:, 'months'] = df.time / float(60*24*30)
        df.loc[:, 'month'] = df.months % 12 + 1

        df.loc[:, 'year'] = df.time / float(60*24*365) + 1

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
        xranges = list([round(x, 3) for x in self.frange(xmin, xmax-self.xsize, self.xslide)])
        yranges = list([round(y, 3) for y in self.frange(ymin, ymax-self.ysize, self.yslide)])
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

    def train(self):
        for i, window in enumerate(self.windows):
            print 'Training Model: {} of {}'.format(i, len(self.windows))
            (x1, y1), (x2, y2) = window
            file_name = file_name_str.format(x1, y1, x2, y2)
            if os.path.isfile(file_name):
                print 'Already Trained'
                continue

            model = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.n_jobs)
            print 'Training Model: {}'.format(model)
            (x1, y1), (x2, y2) = window
            model_df = self.df[(self.df[self.xcol] >= x1) & (self.df[self.xcol] <= x2) &
                               (self.df[self.ycol] >= y1) & (self.df[self.ycol] <= y2)]

            model_df = model_df.sort_values('row_id').set_index('row_id')
            values = model_df['place_id']
            model_df = model_df[self.features]

            if not len(model_df):
                import pdb; pdb.set_trace()

            model.fit(model_df, values)
            file_name = file_name_str.format(x1, y1, x2, y2)
            joblib.dump(model, file_name, compress=5, )

            model_df = None; del model_df; gc.collect()
            model = None; del model; gc.collect()

    def load_model(self, window):
        (x1, y1), (x2, y2) = window
        file_name = file_name_str.format(x1, y1, x2, y2)
        model = joblib.load(file_name)
        return model

    def round_off(self, x):
        return round(x, 3)

    def predict(self, df, test=False):
        df = df.sort_values('row_id')
        self.result_set = {}
        self.dr = {}

        if test:
            self.expected = dict(zip(df.row_id, df.place_id))
            self.order_expected = df.place_id

        self.mod_df(df)

        df.loc[:, 'x1'] = df.x.apply(self.find_x_window)
        df.loc[:, 'x2'] = df.x1 + self.xsize
        df.loc[:, 'x2'] = df.x2.apply(self.round_off)
        df.loc[:, 'y1'] = df.y.apply(self.find_y_window)
        df.loc[:, 'y2'] = df.y1 + self.ysize
        df.loc[:, 'y2'] = df.y2.apply(self.round_off)

        out_range = df[(df.x < df.x1) | (df.x > df.x2) | (df.y < df.y1) | (df.y > df.y2)]
        if len(out_range):
            print 'Error in window generation'
            import pdb; pdb.set_trace()

        for i, window in enumerate(self.windows):
            print 'Predicting Model: {} of {}'.format(i, len(self.windows))
            model = self.load_model(window)
            (x1, y1), (x2, y2) = window
            wdf = df[(df.x1 == x1) & (df.x2 == x2) & (df.y1 == y1) & (df.y2 == y2)]

            wdf = wdf.sort_values('row_id').set_index('row_id')
            wdf = wdf[self.features]

            if not len(wdf):
                import pdb; pdb.set_trace()
                continue

            # Making single predictions for this set
            predictions = model.predict_proba(wdf)
            best_fit = model.predict(wdf)
            self.dr.update(dict(zip(wdf.index, best_fit)))

            # Making proba predictions for this set
            for index in xrange(len(wdf)):
                res = {}
                place_list = []
                row_id = wdf.index[index]
                prediction = predictions[index]

                indices = np.argsort(prediction)[-3:][::-1]
                places = model.classes_[indices]
                probabilities = np.sort(prediction)[-3:][::-1]

                p1, p2, p3 = probabilities
                assert p1 >= p2 >= p3

                place_list.append(places[0])
                p2_appended = False
                if p1 - p2 <= 0.2:
                    place_list.append(places[1])
                    p2_appended = True
                if p2_appended:
                    if p2 - p3 <= 0.1:
                        place_list.append(places[2])

                res[row_id] = place_list
                self.result_set.update(res)

            model = None; del model; gc.collect()

        if test:
            self.actual = (self.result_set[x][0] for x in sorted(self.result_set.keys()))

        with open('result.json', 'w') as f1:
            pruned = {k: v[0] for k,v in self.result_set.iteritems()}
            f1.write(json.dumps(pruned)) 
        with open('result2.json', 'w') as f2:
            f2.write(json.dumps(self.dr))

        return self.result_set

    def write_result(self, file_name):
        with open(out_file_name, 'w') as outfile:
            csv_writer = csv.writer(outfile, delimiter=',')
            for row_id, place_list in self.result_set.iteritems():
                place_str = ' '.join((str(x) for x in place_list))
                csv_writer.writerow([row_id, place_str])

        with open(out_file_name2, 'w') as outfile2:
            csv_writer = csv.writer(outfile2, delimiter=',')
            for row_id, place in self.dr.iteritems():
                csv_writer.writerow([row_id, place])

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
        m = pd.Series((self.dr[x] for x in sorted(self.dr.keys())))
        print 'Simple Score Multi Pred: {}'.format((sum(o == n) / float(len(o))) * 100)
        print 'Simple Score One Pred: {}'.format((sum(o == m) / float(len(o))) * 100)
        print 'Simple Score: {}'.format((correct_count/float(total_count)) * 100)
        print 'Mean Score: {}'.format((mean_score/float(total_count)) * 100)
        return mean_score


def run(xsize, ysize, xstep, ystep, n_estimators, n_jobs):
    try:
        os.remove(out_file_name)
    except:
        pass
    try:
        os.remove(out_file_name2)
    except:
        pass
    print 'Loading DataFrame'
    df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
    df_test = pd.read_csv('Kaggle_Datasets/Facebook/test.csv')

    # df_train = df_train.loc[(df_train.x <= 1) & (df_train.y <= 1), :]
    # df_test = df_test.loc[(df_test.x <= 1) & (df_test.y <= 1), :]

    print 'Splitting train and test data'
    train, test = train_test_split(df_train, test_size=0.2, random_state=1)
    #train, cv = train_test_split(train, test_size=0.25, random_state=2)

    df_train = None; del df_train; gc.collect()

    print 'Initializing PredictionModel class'
    pred_model = MultiPredictionModel(train, xsize, ysize, xstep, ystep, 'x', 'y', n_estimators, n_jobs)
    print 'Init done'
    print pred_model.windows

    print 'Training Model'
    pred_model.train()
    print 'Done Training'

    print 'Predicting on test data'
    print pred_model.predict(test, test=True)
    print 'Done predicting'

    print 'Scoring Data'
    pred_model.score()
    print 'Done Scoring'

    print 'Predicting on read data'
    print pred_model.predict(df_test)
    print 'Done predicting'   

    print 'Print Writing Results'
    pred_model.write_result()    
    print 'Done writing results'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xsize", help="Size of X", type=float)
    parser.add_argument("ysize", help="Size of Y", type=float)
    parser.add_argument("xstep", help="Step of X", type=float)
    parser.add_argument("ystep", help="Step of Y", type=float)
    parser.add_argument("n_estimators", help="Number of estimators", type=int)
    parser.add_argument("n_jobs", help="Number of parallel jobs", type=int)
    args = parser.parse_args()
    xsize = round(args.xsize, 3)
    ysize = round(args.ysize, 3)
    xslide = round(args.xstep, 3)
    yslide = round(args.ystep, 3)
    n_estimators = args.n_estimators
    n_jobs = args.n_jobs
    print run(xsize, ysize, xslide, yslide, n_estimators, n_jobs)
