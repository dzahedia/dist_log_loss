from scipy import stats
from math import log

def dist_log_loss(y_true, y_pred, labels=[]):
    """Log loss, aka logistic loss or cross-entropy loss.
    
    For a single sample with true label yt in {0,1} and
    estimated probability yp that yt = 1, the log loss is

        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
    
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.

    y_pred : array-like of float, shape = (n_samples, n_classes) 
        Predicted probabilities      
   

    labels : array-like, optional (default=None)
        If not provided, labels will be inferred from y_true. 

    Returns
    -------
    loss : float
    """
    
    losses = []
    if len(labels):
        for tr, prob in zip(y_true,y_pred):
            pt = prob[labels.index(tr)]
            losses.append(-log(pt))
    else:
        fit_labels = sorted(list(set(y_true)))
        for tr, prob in zip(y_true,y_pred):
            pt = prob[fit_labels.index(tr)]
            losses.append(-log(pt))
    d = stats.describe(losses)
    return {'mean':d.mean, 'variance': d.variance, 'skewness':d.skewness, 'kurtosis':d.kurtosis}
