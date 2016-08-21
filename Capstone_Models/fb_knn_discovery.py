import logging
from copy import copy
import json
import csv
import operator
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

logging.basicConfig(filename='fb_knn.log', level=logging.DEBUG)


def prepare_data(df, conf, features):
    """
    Feature engineering and computation of the grid.
    """
    # Creating the grid
    size_x = 10.0 / conf['n_cell_x']
    size_y = 10.0 / conf['n_cell_y']
    eps = 0.00001
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df['grid_cell'] = pos_y * conf['n_cell_x'] + pos_x

    # Feature engineering

    df.x = df.x.values * conf['x']
    df.y = df.y.values * conf['y']
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                               for mn in df.time.values)
    df['hour'] = d_times.hour * conf['hour']
    df['weekday'] = d_times.weekday * conf['weekday']
    df['day'] = (d_times.dayofyear * conf['day']).astype(int)
    df['month'] = d_times.month * conf['month']
    df['year'] = (d_times.year - 2013) * conf['year']
    df['log2accu'] = np.log2(df.accuracy) * conf['log2accu']

    df = df.drop(['time'], axis=1)
    df = df.drop(['accuracy'], axis=1)

    return df


def process_one_cell(df_train, df_test, grid_id, th):
    """
    Classification inside one grid cell.
    """
    # Working on df_train
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    # Working on df_test
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    row_ids = df_cell_test.index

    # Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'grid_cell'], axis=1).values.astype(int)
    X_test = df_cell_test.drop(['grid_cell'], axis=1).values.astype(int)

    # Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=conf['neighbours'], weights='distance',
                               metric='manhattan')
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:, ::-1][:, :3])
    return pred_labels, row_ids


def process_grid(df_train, df_test, conf):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """
    preds = np.zeros((df_size, 3), dtype=int)
    n_cells = conf['n_cell_x'] * conf['n_cell_y']
    for g_id in range(n_cells):
        if g_id % 100 == 0:
            print('iter: %s' % (g_id))

        # Applying classifier to one grid cell
        pred_labels, row_ids = process_one_cell(df_train, df_test, g_id, conf['th'])

        # Updating predictions
        preds[row_ids] = pred_labels

    print('Computing Score...')
    # Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    return df_aux
    # Concatenating the 3 predictions for each sample
    # ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

    # Writting to csv
    # ds_sub.name = 'place_id'
    # ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')


def compute_score(results, actuals):
    res = results.loc[actuals.index, :]
    r1 = res.l1.astype(np.int)
    r2 = res.l2.astype(np.int)
    r3 = res.l3.astype(np.int)
    s1 = r1 == actuals
    s2 = r2 == actuals
    s3 = r3 == actuals
    import pdb; pdb.set_trace()
    return sum(s1*1 + s2*0.5 + s3*0.3)/float(len(actuals))


if __name__ == '__main__':
    print('Loading data ...')
    df = pd.read_csv('../Kaggle_Datasets/Facebook/train.csv',
                     usecols=['row_id', 'x', 'y', 'time', 'accuracy', 'place_id'],
                     index_col=0)

    # df_test = pd.read_csv('../Kaggle_Datasets/Facebook/test.csv',
    #                       usecols=['row_id', 'x', 'y', 'time'],
    #                       index_col=0)

    # df_train = df_train[(df_train.x < 0.5) & (df_train.y < 0.5)]
    # df_test = df_test[(df_test.x < 0.5) & (df_test.y < 0.5)]
    # Defining the size of the grid
    df_size = len(df)
    # df = df.sample(500000, random_state=78)
    df_train, df_test = train_test_split(df, random_state=77, test_size=0.2)
    df_train, df_cv = train_test_split(df_train, random_state=88, test_size=0.2)
    df = None

    features = ['x', 'y', 'hour', 'weekday', 'day', 'month', 'year', 'log2accu']

    conf = {
        'neighbours': 25,
        'n_cell_x': 20,
        'n_cell_y': 40,
        'th': 5,
        'x': 500,
        'y': 1000,
        'hour': 4,
        'weekday': 3,
        'day': 1. / 22.,
        'month': 2,
        'year': 10,
        'log2accu': 100,
    }

    result = {}

    # BaseLine
    actuals = df_test.place_id
    dftr = prepare_data(df_train, conf, features)
    print('Baseline')
    dftst = prepare_data(df_test, conf, features)
    dftst = dftst.drop('place_id', 1)
    # Solving classification problems inside each grid cell
    r = process_grid(dftr, dftst, conf, )

    score = compute_score(r, actuals)
    logging.info('Baseline: {}'.format(score))
    result['BaseLine'] = score

    for var, val in conf.items():
        for v in (val / 2.0, val * 2.0):
            logging.info('Varying: {} to {}'.format(var, val))
            print ('Varying: {} to {}'.format(var, val))
            conf_mod = copy(conf)
            conf_mod[var] = v
            # Try doubling/halving and look at the impact. Proceed accordingly
            logging.info('Varying: {} to {}'.format(var, val))
            print ('Varying: {} to {}'.format(var, val))
            print('Preparing train data')
            dftr = prepare_data(df_train, conf_mod, features)

            actuals = df_test.place_id
            logging.info('Varying: {} to {}'.format(var, val))
            print('Preparing test data')
            dftst = prepare_data(df_test, conf_mod, features)
            dftst = dftst.drop('place_id', 1)
            # Solving classification problems inside each grid cell
            r = process_grid(dftr, dftst, conf_mod, )

            score = compute_score(r, actuals)
            result[str(conf_mod)] = score

            logging.info('Varying: {} to {}'.format(var, val))
            logging.info('Score: {}\nConf: {}'.format(score, conf_mod))
            print ('Score: {}\nConf: {}'.format(score, conf_mod))

    sorted_score = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    with open('feature_result.csv', 'w') as f:
        writer = csv.writer(f)
        for keys, score in sorted_score:
            writer.writerow([score, keys])

    logging.info('Max Score: {}'.format(sorted_score[0]))
