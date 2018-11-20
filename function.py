from scipy import stats
from math import log

def dist_log_loss(true, prob, lab_ord=[]):
    """Description to come"""
    
    losses = []
    if len(lab_ord):
        for tr, prob in zip(true,prob):
            pt = prob[lab_ord.index(tr)]
            losses.append(-log(pt))
    else:
        labord = sorted(list(set(true)))
        for tr, prob in zip(true,prob):
            pt = prob[labord.index(tr)]
            losses.append(-log(pt))
    d = stats.describe(losses)
    return {'mean':d.mean, 'variance': d.variance, 'skewness':d.skewness, 'kurtosis':d.kurtosis}
