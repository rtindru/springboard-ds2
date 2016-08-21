import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
# df_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/fbdataset/test.csv')


class SinglePredictionModel(object):

    def __init__(self, df):
        self.df = df
        self.xmax = self.df.x.max()
        self.ymax = self.df.y.max()
        self.model = None
        self.features = ['x', 'y', 'accuracy', 'hour', 'day', 'week', 'month', 'year']
        self.expected = None
        self.actual = None

    def mod_df(self, df):
        df.loc[:, 'hours'] = df.time / float(60)
        df.loc[:, 'hour'] = df.hours % 24

        df.loc[:, 'days'] = df.time / float(60*24)
        df.loc[:, 'day'] = df.days % 7

        df.loc[:, 'weeks'] = df.time / float(60*24*7)
        df.loc[:, 'week'] = df.weeks % 52

        df.loc[:, 'months'] = df.time / float(60*24*30)
        df.loc[:, 'month'] = df.months % 12

        df.loc[:, 'year'] = df.time / float(60*24*365)

    def train(self):
        self.model = RandomForestClassifier(n_jobs=-1, warm_start=True)

        self.mod_df(self.df)
        
        train_df = self.df.loc[:, self.features]
        values = self.df.loc[:, 'place_id']
        self.model.fit(train_df, values)

    def predict(self, df):
        self.mod_df(df)
        wdf = df.sort_values('row_id').set_index('row_id')        
        self.expected = wdf.place_id
        wdf = wdf.loc[:, self.features]

        predictions = self.model.predict(wdf)
        self.actual = predictions
        return dict(zip(wdf.index, predictions))

    def score(self):
        expect = pd.Series(self.expected)
        actual = pd.Series(self.actual)
        return (sum(expect == actual)/float(len(self.expected))) * 100


def run():
    print 'Loading DataFrame'
    df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
    # df_train = df_train.loc[(df_train.x <= 0.5) & (df_train.y <= 0.5), :]

    print 'Splitting train and test data'
    train, test = train_test_split(df_train, test_size=0.2)

    print 'Initializing PredictionModel class'
    pred_model = SinglePredictionModel(df=train)
    print 'Init done'

    print 'Training Model'
    pred_model.train()
    print 'Done Training'

    print 'Predicting on test data'
    predictions = pred_model.predict(test)
    print predictions
    print test.place_id
    print 'Done predicting'

    score = pred_model.score()
    print 'Score: {}'.format(score)
    return score
    
run()
