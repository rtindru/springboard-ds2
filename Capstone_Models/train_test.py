import pandas as pd
import numpy as np
import scipy
import matplotlib
import seaborn as sns
from sklearn.cross_validation import train_test_split

df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
df_test = pd.read_csv('Kaggle_Datasets/Facebook/test.csv')


class PredictionModel():
    def __init__(self, df, xsize=1, ysize=0.5, xslide=0.5, yslide=0.25, xcol='x', ycol='y'):
        self.df = df
        self.xsize = xsize
        self.ysize = ysize
        self.xslide = xslide
        self.yslide = yslide
        self.xcol = xcol
        self.ycol = ycol
        
        self.windows = self.generate_windows()
        self.slices = self.slice_df()
    
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
            result.append(((x1, y1), (x1+self.xsize, y1+self.ysize)))
        
        return result

    def slice_df(self):
        slices = {}
        for window in self.windows:
            slices[window] = ModelStore(self.df, window, self.xcol, self.ycol)

        return slices

    
class ModelStore():
    def __init__(self, df, window, xcol, ycol):
        self.window = window
        (self.x1, self.y1), (self.x2, self.y2) = self.window 
        self.df = df[(df[xcol] >= self.x1) & (df[xcol] <= self.x2) & (df[ycol] >= self.y1) & (df[ycol] <= self.y2)]
        self.unique_place_count = len(self.df.place_id.unique())
        self.model = None
        self.df['hours'] = self.df.time / 60.0
        self.df['days'] = self.df.time / (60*24)
        self.df['hours_cycle'] = self.df.hours % 24
        self.df['days_cycle'] = self.df.days % 7
	self.total_count = len(self.df)
        
    def train(self, model='logistic'):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=5)  # x, y, accuracy, hours_cycle, days_cycle
        self.train_df = self.df.sort_values('row_id')[['x', 'y', 'accuracy', 'hours_cycle', 'days_cycle']]
        self.values = self.df.sort_values('row_id')['place_id']
        self.model.fit(self.train_df, self.values)


    def describe(self):
	return '{}: {}, {}'.format(self, window, self.total_count, self.unique_place_count)

df_under_1 = df_train[(df_train.x <= 1.0) & (df_train.y <= 1.0)]
train, test = train_test_split(df_under_1, test_size = 0.2)

pred_model = PredictionModel(df=train)
print pred_model.slices
for window, model in pred_model.slices.iteritems():
    print model.describe()
    model.train()

