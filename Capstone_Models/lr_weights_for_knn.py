# This code helps to find out feature multipliers for KNN.
# This is shown using some features derived by me but this method can be extended for other features as well.
# One needs to derive his own features and then apply similar approach to get the correct weights.


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

recent_train = pd.read_csv('../Kaggle_Datasets/Facebook/train.csv')

def prepare_data(df, n_cell_x=None, n_cell_y=None):
    """
    Feature engineering and computation of the grid.
    """
    #Creating the grid
    # size_x = 10.0 / n_cell_x
    # size_y = 10.0 / n_cell_y
    # eps = 0.00001
    # xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    # ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    # pos_x = (xs / size_x).astype(np.int)
    # pos_y = (ys / size_y).astype(np.int)
    # df['grid_cell'] = pos_y * n_cell_x + pos_x

    #Feature engineering
    # fw = [500, 1000, 4, 3, 1./22., 2, 10] #feature weights (black magic here)
    df.x = df.x.values
    df.y = df.y.values
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                               for mn in df.time.values)
    df['hour'] = d_times.hour
    df['weekday'] = d_times.weekday
    df['day'] = (d_times.dayofyear).astype(int)
    df['month'] = d_times.month
    df['year'] = (d_times.year - 2013)
    df['log2accu'] = np.log2(df.accuracy)

    # df = df.drop(['time'], axis=1)
    return df

# select a single x_y_grid at random
recent_train = recent_train[
    (recent_train["x"] > 4.5) & (recent_train["x"] < 5) & (recent_train["y"] > 2) & (recent_train["y"] < 2.3)]

use_features = ['x', 'y', 'hour', 'weekday', 'day', 'month', 'year', 'log2accu', 'accuracy', 'time', 'place_id']
features = ['x', 'y', 'hour', 'weekday', 'day', 'month', 'year', 'log2accu', 'accuracy', 'time']
# derive some features
# recent_train["x"], recent_train["y"] = recent_train["x"] * 1000, recent_train["y"] * 1000
# recent_train["hour"] = recent_train["time"] // 60
# recent_train["hour_of_day"] = recent_train["hour"] % 24 + 1
#
# recent_train["day"] = recent_train["hour"] // 24
# recent_train["day_of_week"] = recent_train["day"] % 7 + 1
#
# recent_train["month"] = recent_train["day"] // 30 + 1
# recent_train["month_of_year"] = (recent_train["month"] - 1) % 12 + 1
#
# recent_train["sine"] = np.sin(2 * np.pi * recent_train["hour_of_day"] / 24)
# recent_train["cos"] = np.cos(2 * np.pi * recent_train["hour_of_day"] / 24)
#
# recent_train["year"] = recent_train["day"] // 365 + 1
recent_train = prepare_data(recent_train)
recent_train = recent_train.loc[:, use_features]
print("recent_train created")

# creating arbitrary test
test = recent_train.sample(axis=0, frac=0.05)
print ("selected_part and test created")
# features = ["x", "y", "hour_of_day", "day_of_week", "month_of_year", "year", "sine", "cos", "accuracy"]
constant = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print (len(test))

colname = str(features)
test[colname] = list
index = test.index
test["done"] = 0
for i in index:
    # manhattan distance between train and test[i]
    new_ld = abs(recent_train[features] - test.loc[i][features])
    new_ld = new_ld.drop(i)
    new_ld["target"] = (recent_train["place_id"] != test.loc[i]["place_id"]) + 0
    # select 100 nearest points based on x+2y distance
    new_ld["x+y"] = (new_ld["x"]) + (2 * new_ld["y"])
    new_ld = new_ld.sort("x+y")[0:100]
    true = new_ld[new_ld["target"] == 0]
    false = new_ld[new_ld["target"] != 0]
    # check for skewness
    if (len(true) < 20) | (len(false) < 20):
        print ("skipped test sample -", i)
        continue
    # get the multipliers which can distinguish between 0 and 1
    lr.fit(new_ld[features], new_ld["target"])
    test.set_value(i, colname, np.maximum(constant, lr.coef_.ravel()))
    #    actual_test.set_value(i,colname,lr.coef_.ravel())
    test.set_value(i, "done", 1)
    print ("done test sample", i)

# average or sum all the multipliers to get overall multiplier
actual_test2 = test[test["done"] == 1]
final_weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
for lists in actual_test2[colname]:
    final_weights = final_weights + lists

print (features)
print ("corresponding weights")
print (final_weights)
# Results
['x', 'y', 'hour', 'weekday', 'day', 'month', 'year', 'log2accu', 'accuracy', 'time']
[2.78081211e-02, 2.00890984e-02, 2.02405885e+01, 1.03837023e+01, 1.87224053e+00, 3.94438968e+00, 5.04880080e+00, 1.28098213e+01, 5.64821133e+00, 2.38557461e-03]
