'''
Calculates the p-values using bootstrapping and random forest
'''

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics
import numpy as np

#random forest params
NUMB_ESTIMATORS=100
MIN_SAMPLES_LEAF=10

class pValInfo():
    '''
    hold permuted list and non-permuted initial val
    returns p-value
    '''

    def __init__(self, col):
        """
        param cpol: column we are operating on
        """
        self.col = col
        self.correct_pred = []
        self.permuted_preds = []

    def get_pval(self):
        '''
        returns the percentage of values in the permuted preds that are larger than the
        non permuted pred
        '''
        # if either of the above are empty throws exception
        n = sum(i > self.col for i in self.permuted_preds)
        return (n / len(self.permuted_preds))

def get_all_pvals(all_columns, trn, trn_y, tst, numb_iter):
    '''
    Get all the column p-values
    :param all_columns:
    :param trn:
    :param trn_y:
    :param tst:
    :param numb_iter:
    :return:
    '''
    res = []
    for col in all_columns:
        res.append(get_col_pval(col, trn, trn_y, tst, numb_iter))

def get_col_pval(col, trn, trn_y, tst, numb_iter):
    '''
    Get a single columns p-value
    :param col: the column we are operating on
    :param trn:
    :param trn_y:
    :param tst:
    :param numb_iter:  how many AMEs to generate on permuted data, used to produce prediction normal distribution
    :return: numbiter AME calcs
    '''
    val = pValInfo()    #holder
    val.correct_preds = get_AME(col, trn, trn_y, tst, numb_iter=numb_iter)

    #permute column
    cpy=trn[col].copy() #save for later
    trn[col]=np.random.permutation(trn[col].values)

    val.permuted_preds = get_AME(col, trn, trn_y, tst, numb_iter=numb_iter)

    #replace permuted column
    trn[col] = cpy

    return val

def get_AME(col, trn, trn_y, tst, numb_iter):
    '''
     Returns numb_iter AME calculations
    :param col: the column we are operating on
    :param trn:
    :param trn_y:
    :param tst:
    :param numb_iter:  how many AMEs to generate on permuted data, used to produce prediction normal distribution
    :return: numbiter AME calcs
    '''
    res = []
    for _ in range(numb_iter):
        # create and train a random forest object
        m_rf = RandomForestClassifier(n_estimators=NUMB_ESTIMATORS, n_jobs=-1, oob_score=True, max_features='auto',
                                      min_samples_leaf=MIN_SAMPLES_LEAF, verbose=False);
        _ = m_rf.fit(trn, trn_y)

        #get probabilities on test set
        prob_orig = ((m_rf.predict_proba(tst))[:,0]).sum()/len(tst)

        #now lets jump by 1 std dev up (or toggle if binary)
        tst1 = tst.copy()

        if tst1[col].nunique() == 2:
            min = tst1[col].min()
            max = tst1[col].max()

            tst1[col].apply(lambda x: min if x == max else max)
        else:
            tst1[col] += 1

        prob_new = ((m_rf.predict_proba(tst1))[:,0]).sum()/len(tst1)

        res.append(prob_new - prob_orig)

    return res





