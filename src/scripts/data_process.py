from src.data.DataLoader import DataLoader
from src.data.DataProcessor import DataProcessor

def main():
    X_train, X_test, y_train, y_test = DataLoader.load()

    X_train = DataProcessor.process(X_train)
    X_test = DataProcessor.process(X_test)

    X_train.loc[:, "y"] = y_train
    X_test.loc[:, "y"] = y_test

    X_train.to_csv("data/trainset.csv", index=False)
    X_test.to_csv("data/testset.csv", index=False)
    return

if __name__ == "__main__":
    main()