import math

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

x_bins = 1000
y_bins = 1500
time_bins = 6
x_std = 0.01
y_std = 0.005
grid_size = 10.0


class LocationTimeHistogramClassifier(object):
    def __init__(self, classifier, config):
        self.data_dict = {}
        self.model_dict = {}
        self.place_dict = {}
        self.classifier = classifier
        self.config = config

    def add_place(self, x, y, t, place_id):
        idx = self.get_idx(x, y, t)
        self.add_place_id(idx, place_id)
        if idx in self.data_dict:
            df = self.data_dict[idx]
        else:
            df = pd.DataFrame()
            self.data_dict[idx] = df
        dist = self.dist_from_center(x, y)
        df_val = pd.DataFrame([(x, y, dist, place_id), ], columns=['x', 'y', 'dist', 'place_id'])
        df = df.append(df_val)
        self.data_dict[idx] = df

    def add_place_id(self, idx, place_id):
        if idx in self.place_dict:
            place_set = self.place_dict[idx]
        else:
            place_set = set()
            self.place_dict[idx] = place_set
        place_set.add(place_id)
        self.place_dict[idx] = place_set

    def get_x_bin(self, x):
        return math.floor((x * x_bins) / grid_size)

    def get_y_bin(self, y):
        return math.floor((y * y_bins) / grid_size)

    def get_time_bin(self, time):
        return int(math.ceil(time % (24 * 60) / ((24 * 60) / time_bins)))

    def get_idx(self, x, y, t):
        x_bin = self.get_x_bin(x)
        y_bin = self.get_y_bin(y)
        time_bin = self.get_time_bin(t)
        return int((int(x_bin * y_bins + y_bin) * time_bins) + time_bin)

    def retrain(self, idx):
        if idx in self.model_dict:
            model = self.model_dict[idx]
        else:
            model = self.classifier(**self.config)
            self.model_dict[idx] = model
        data = self.data_dict[idx]
        X = data.drop('place_id', axis=1)
        y = data.place_id
        model.fit(X, y)
        self.model_dict[idx] = model

    def predict(self, x, y, t):
        idx = self.get_idx(x, y, t)
        try:
            model = self.model_dict[idx]
        except:
            return [0, 0, 0], [0, 0, 0]
        dist = self.dist_from_center(x, y)
        df = pd.DataFrame([(x, y, dist)], columns=('x', 'y', 'dist'))
        pred = model.predict_proba(df)[0]
        try:
            top_values = np.sort(pred)[-5:][::-1]
            top_indices = np.argsort(pred)[-5:][::-1]
            top_classes = model.classes_[top_indices]
        except:
            return [0, 0, 0], [0, 0, 0]
        return top_classes, top_values

    def train_all(self):
        idxs = self.data_dict.keys()
        for idx in idxs:
            self.retrain(idx)

    def get_candidates(self, x, y, t):
        idx = self.get_idx(x, y, t)
        return self.place_dict[idx]

    def get_xbc(self, xb):
        return (xb + 0.5) * grid_size / x_bins

    def get_ybc(self, yb):
        return (yb + 0.5) * grid_size / y_bins

    def dist_from_center(self, x, y):
        xb, yb = self.get_x_bin(x), self.get_y_bin(y)
        xbc, ybc = self.get_xbc(xb), self.get_xbc(yb)
        zx = (x - xbc) / x_std
        zy = (y - ybc) / y_std
        return math.exp(-(zx * zx + zy * zy) / 2.0)


df_train = pd.read_csv('../../Kaggle_Datasets/Facebook/train_set.csv')
df_test = pd.read_csv('../../Kaggle_Datasets/Facebook/test_set.csv')
df_train = df_train.drop('Unnamed: 0', axis=1)
df_test = df_test.drop('Unnamed: 0', axis=1)
df_train = df_train.drop('row_id', axis=1)
df_test = df_test.drop('row_id', axis=1)
df_train['time_bin'] = np.floor((df_train.time / 60) % 6).astype(int) + 1
df_test.head()


a = LocationTimeHistogramClassifier(MultinomialNB, config={})
for index, row in df_train.iterrows():
    a.add_place(row.x, row.y, row.time_bin, row.place_id)
a.train_all()

cruelscore = 0
safescore = 0
for index, row in df_test.iterrows():
    results, vals = a.predict(row.x, row.y, row.time_bin)
    try:
        if row.place_id == results[0]:
            safescore += 1
        elif row.place_id == results[1]:
            safescore += 0.66
        elif row.place_id == results[2]:
            safescore += 0.33
    except:
        pass
    if row.place_id == results[0]:
        cruelscore += 1

print (cruelscore/float(len(df_test))) * 100
print (safescore/float(len(df_test))) * 100