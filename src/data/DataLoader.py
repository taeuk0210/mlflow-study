from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        pass

    @classmethod    
    def load(cls, test_size=0.3, random_state=42):
        dataset = load_breast_cancer(as_frame=True)

        X = dataset.data
        y = dataset.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state)
        return X_train, X_test, y_train, y_test