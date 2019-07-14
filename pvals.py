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
        n = sum(i > self.correct_pred for i in self.permuted_preds)
        val= (n / len(self.permuted_preds)).item(0)

        #cheesy but it doesn't matter if they are all above or below
        val=val if val<0.5 else (1-val)
        return val

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
        trncpy = trn.copy()
        res.append(get_col_pval(col, trncpy, trn_y, tst, numb_iter))
    return res

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
    val = pValInfo(col)    #holderS

    val.correct_pred = get_AME(col, trn, trn_y, tst, numb_iter=1)
    print(f'correct get_AME for column {col} is {val.correct_pred[0]}')

    #permute column
    trn[col]=np.random.permutation(trn[col].values)

    print(f'get permuted get_AME for column {col}')
    val.permuted_preds = get_AME(col, trn, trn_y, tst, numb_iter=numb_iter)

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
    cnt=0
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
            tst1[col] += 1.0

        prob_new = ((m_rf.predict_proba(tst1))[:,0]).sum()/len(tst1)

        res.append(prob_new - prob_orig)

        cnt=(cnt+1)%10
        if (cnt==0):
            print(".",end='')
    print()
    return res





