import pandas as pd
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

# df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
# df_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/fbdataset/test.csv')


def run():
    print 'Loading DataFrame'
    df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
    # df_train = df_train.loc[(df_train.x <= 0.5) & (df_train.y <= 0.5), :]

    print 'Splitting train and test data'
    train, test = train_test_split(df_train, test_size=0.2)

    df = train
    features = ['x', 'y', 'accuracy', 'hour', 'day', 'week', 'month', 'year']
    
    df.loc[:, 'hours'] = df.time / float(60)
    df.loc[:, 'hour'] = df.hours % 24

    df.loc[:, 'days'] = df.time / float(60*24)
    df.loc[:, 'day'] = df.days % 7

    df.loc[:, 'weeks'] = df.time / float(60*24*7)
    df.loc[:, 'week'] = df.weeks % 52

    df.loc[:, 'months'] = df.time / float(60*24*30)
    df.loc[:, 'month'] = df.months % 12

    df.loc[:, 'year'] = df.time / float(60*24*365)

    model = RandomForestClassifier(n_jobs=1, warm_start=True)
    
    train_df = df.loc[:, features]
    values = df.loc[:, 'place_id']
    
    print 'Fitting Model'
    model.fit(train_df, values)

    wdf = test.sort_values('row_id').set_index('row_id')        
    expected = wdf.place_id
    wdf = wdf.loc[:, features]

    predictions = model.predict(wdf)
    actual = predictions
    print dict(zip(wdf.index, predictions))

    expect = pd.Series(expected)
    actual = pd.Series(actual)
    print (sum(expect == actual)/float(len(expected))) * 100
