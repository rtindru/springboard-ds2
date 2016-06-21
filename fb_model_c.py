import pandas as pd
from sklearn.cross_validation import train_test_split

df_train = pd.read_csv('Kaggle_Datasets/Facebook/train.csv')
#df_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/fbdataset/test.csv')


class PredictionModel():
    
    def __init__(self, df, xsize=1, ysize=0.5, xslide=0.5, yslide=0.25, xcol='x', ycol='y'):
        self.df = df
        self.xsize = xsize
        self.ysize = ysize
        self.xslide = xslide
        self.yslide = yslide
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = self.df.x.max()
        self.ymax = self.df.y.max()
        
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
    
    def find_best_window(self, df):
        x1, y1 = self.find_x_window(x), self.find_y_window(y)
        x2, y2 = x1+self.xsize, y1+self.ysize

        try:
            assert x1 <= x <= x2
            assert y1 <= y <= y2
        except:
            import pdb; pdb.set_trace()

        return ((x1, y1), (x2, y2))
    
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
    	for window, model in self.slices.iteritems():
    	    print 'Training Model: {}'.format(model)
            (x1, y1), (x2, y2) = window
            model_df = self.df[(self.df[self.xcol] >= x1) & (self.df[self.xcol] <= x2) & (self.df[self.ycol] >= y1) & (self.df[self.ycol] <= y2)]
    	    model.train(model_df)
            del model_df
    
    def predict(self, df):
        self.expected = df.sort_values('row_id')['place_id']
        result_set = {}
        df['x1'] = df.x.apply(self.find_x_window)
        df['x2'] = df.x1 + self.xsize
        df['y1'] = df.y.apply(self.find_y_window)
        df['y2'] = df.y1 + self.ysize

        for window, model in self.slices.iteritems():
            (x1, y1), (x2, y2) = window
            wdf = df[(df.x1 == x1) & (df.x2 == x2) & (df.y1 == y1) & (df.y2 == y2)]
            res = model.predict(wdf)
            result_set.update(res)

        self.actual = [result_set[x] for x in sorted(result_set.keys())]
        return result_set
    
    def score(self):
        expect = pd.Series(self.expected)
        actual = pd.Series(self.actual)
        return (sum(expect == actual) / float(len(self.expected))) * 100
        

class ModelStore():

    def __init__(self, df, window, xcol, ycol):
        self.window = window
        self.xcol = xcol
        self.ycol = ycol
        (self.x1, self.y1), (self.x2, self.y2) = self.window 
        self.unique_place_count = len(df.place_id.unique())
        self.model = None
        self.total_count = len(df)
        
    def __unicode__(self):
        return '{}: {}, {}'.format(self.window, self.total_count, self.unique_place_count)

    def get_self_df(self, df):
        self_df = df
        self_df['hours'] = self_df.time / 60.0
        self_df['days'] = self_df.time / (60*24.0)
        self_df['hours_cycle'] = self_df.hours % 24
        self_df['days_cycle'] = self_df.days % 7
        return self_df
    
    def train(self, df):
        self_df = self.get_self_df(df)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=5)  # x, y, accuracy, hours_cycle, days_cycle
        tdf = self_df.sort_values('row_id').set_index('row_id')
        train_df = tdf[['x', 'y', 'accuracy', 'hours_cycle', 'days_cycle']]
        values = tdf['place_id']
        self.model.fit(train_df, values)

    def predict(self, df):
        wdf = df.sort_values('row_id').set_index('row_id')
        wdf = self.mod_df(wdf)
        wdf = wdf[['x', 'y', 'accuracy', 'hours_cycle', 'days_cycle']]
        return dict(zip(wdf.index, self.model.predict(wdf)))

def run():
    print 'Splitting train and test data'
    train, test = train_test_split(df_train, test_size = 0.2)
    print 'Initializing PredictionModel class'
    pred_model = PredictionModel(df=train)
    print 'Init done'
    print pred_model.slices
    
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
