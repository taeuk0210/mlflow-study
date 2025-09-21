from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        pass

    @classmethod
    def process(cls, X):
        X.iloc[:, :] = StandardScaler().fit_transform(X)
        return X