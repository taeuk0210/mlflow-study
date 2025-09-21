import numpy as np

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc

class ModelTrainer:

    def __init__(self, model):
        self.model = model
        pass

    def train(self, X, y, return_model=True):

        model = clone(self.model)
        model.fit(X, y)

        if return_model:
            return model
        return
    
    @classmethod
    def score(cls, model, X, y, threshlod=0.5):

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = np.array(y_prob > threshlod, dtype=np.int8)

        TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
        p, r, _ = precision_recall_curve(y_true=y, probas_pred=y_prob)
        return dict(
            TN=TN, FP=FP, FN=FN, TP=TP,
            accuracy=accuracy_score(y_true=y, y_pred=y_pred),
            precision = precision_score(y_true=y, y_pred=y_pred),
            recall = recall_score(y_true=y, y_pred=y_pred),
            f1score = f1_score(y_true=y, y_pred=y_pred),
            roc_auc = roc_auc_score(y_true=y, y_score=y_prob),
            pr_auc = auc(r, p)
        )


