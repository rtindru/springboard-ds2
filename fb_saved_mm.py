import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import gc

gc.enable()
# df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
# df_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/fbdataset/test.csv')


class MultiPredictionModel(object):
    
    def __init__(self, df, xsize=0.5, ysize=0.5, xslide=0.5, yslide=0.25, xcol='x', ycol='y'):
        self.df = df
        self.xsize = xsize
        self.ysize = ysize
        self.xslide = xslide
        self.yslide = yslide
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = self.df.x.max()
        self.ymax = self.df.y.max()
        self.features = ['x', 'y', 'accuracy', 'hour', 'day', 'week', 'month', 'year']
        self.mod_df(self.df)
        self.windows = self.generate_windows()

        self.expected = None
        self.actual = None
        self.result_set = {}

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

    def frange(self, x, y, jump):
        while x < y:
            yield x
            x += jump
        yield y

    def generate_windows(self):  
        ranges = []
        result = []
        
        xmin, xmax = self.df.x.min(), self.df.x.max()
        ymin, ymax = self.df.y.min(), self.df.y.max()
        xranges = list(self.frange(xmin, xmax-self.xsize, self.xslide))
        yranges = list(self.frange(ymin, ymax-self.ysize, self.yslide))
        ylen = len(yranges)
        for x in xranges:
            subrange = [x] * ylen
            ranges.extend(zip(subrange, yranges))

        for x1, y1 in ranges:
            x2, y2 = x1 + self.xsize, y1 + self.ysize
            result.append(((x1, y1), (x2, y2)))
        
        return result

    def find_x_window(self, x):
        xs = max(0, x - (self.xsize/2.0))
        x0 = 0

        while x0 < xs:
            x0 += self.xslide
        if x0 >= self.xmax - self.xsize: 
            x0 = self.xmax - self.xsize
        return x0
    
    def find_y_window(self, y):
        ys = max(0, y - (self.ysize/2.0))
        y0 = 0

        while y0 < ys:
            y0 += self.yslide
        if y0 >= self.ymax - self.ysize:
            y0 = self.ymax - self.ysize
        return y0

    def train(self):
        for window in self.windows:
            model = RandomForestClassifier(n_estimators=len(self.features))
            print 'Training Model: {}'.format(model)
            (x1, y1), (x2, y2) = window
            model_df = self.df[(self.df[self.xcol] >= x1) & (self.df[self.xcol] <= x2) & (self.df[self.ycol] >= y1) & (self.df[self.ycol] <= y2)]
            model_df = model_df.sort_values('row_id').set_index('row_id')
            values = model_df['place_id']
            model_df = model_df[self.features]

            model.fit(model_df, values)
            file_name = 'mod_{}_{}_{}_{}.pkl'.format(x1, y1, x2, y2)
            joblib.dump(model, file_name)
            del model_df
            del model

    def load_model(self, window):
        (x1, y1), (x2, y2) = window
        file_name = 'mod_{}_{}_{}_{}.pkl'.format(x1, y1, x2, y2)
        model = joblib.load(file_name)
        return model

    def predict(self, df):
        df = df.sort_values('row_id')
        self.expected = df.place_id
        self.mod_df(df)

        df.loc[:, 'x1'] = df.x.apply(self.find_x_window)
        df.loc[:, 'x2'] = df.x1 + self.xsize
        df.loc[:, 'y1'] = df.y.apply(self.find_y_window)
        df.loc[:, 'y2'] = df.y1 + self.ysize
        
        for window in self.windows:
            model = self.load_model(window)
            (x1, y1), (x2, y2) = window
            wdf = df[(df.x1 == x1) & (df.x2 == x2) & (df.y1 == y1) & (df.y2 == y2)]
            wdf = wdf.sort_values('row_id').set_index('row_id')
            wdf = wdf[self.features]
            predictions = model.predict(wdf)

            del model; model = None
            gc.collect()

            res = dict(zip(wdf.index, predictions))
            self.result_set.update(res)

            del res; res = None
            gc.collect()

        self.actual = [self.result_set[x] for x in sorted(self.result_set.keys())]
        return self.result_set
    
    def score(self):
        expect = pd.Series(self.expected)
        actual = pd.Series(self.actual)
        return (sum(expect == actual) / float(len(self.expected))) * 100
        

def run():
    print 'Loading DataFrame'
    df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
    df_train = df_train.loc[(df_train.x <= 0.5) & (df_train.y <= 0.5), :]
    
    print 'Splitting train and test data'
    train, test = train_test_split(df_train, test_size=0.2)
    del df_train

    print 'Initializing PredictionModel class'
    pred_model = MultiPredictionModel(df=train)
    print 'Init done'
    print pred_model.windows
    
    print 'Training Model'
    pred_model.train()
    print 'Done Training'

    print 'Predicting on test data'
    print pred_model.predict(test)
    print 'Done predicting'

    score = pred_model.score()
    print 'Score: {}'.format(score)
    return score
    
run()
