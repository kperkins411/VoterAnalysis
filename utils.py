import pandas as pd
import numpy as np

def replaceNaN(dftarget, cols, val=''):
    """
    Replaces NaNs in dfTarget with '' for every column in cols
    :param df:
    :param cols: list of columns from dfPreds
    :return:
    """
    for col in cols:
        if (col=='index'):
            continue
        # replace NaNs in col of interest with ''
        dftarget.loc[:,col].fillna(val, inplace=True)
    return dftarget


def merge(dfPreds, dfOrig):
    """
    assummes both inputs are dataframes and they both have had reset_index() performed
    to create a new column with orig index numbers called 'index'
    merges dfPreds with dfOrig based on column 'index'.
    uses 'outer' = union join (inner=intersection)
    :param dfPreds: predictions
    :param dfclean: original dataset
    :return: merged dataframe
    """
    # alright put emback together
    dfn = dfPreds.merge(dfOrig,on='index', how='outer')

    #reset the indices
    dfn.set_index('index', inplace=True)

    #sort
    dfn.sort_index(inplace=True)

    return dfn

if __name__=="__main__":
    dforig = pd.DataFrame(np.arange(20).reshape(10, 2), columns=list('ab'))
    dfpreds = pd.DataFrame([[1, 3], [2, 7], [3, 9]], columns=['RF', 'index'])

    dforig.iloc[1, 1] = np.NaN
    dforig.iloc[3, 1] = np.NaN
    dforig.iloc[4, 0] = np.NaN

    # dfnull.reset_index(inplace=True)
    dforig.reset_index(inplace=True)
    dfm = merge(dfpreds, dforig)
    dfm = replaceNaN(dfm, dfpreds.columns)
    dfm = replaceNaN(dfm, dfm.columns)

