import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
    
# DATA SPLIT
def create_folds(df, CFG=None):
    df = df.copy()
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df["bins"] = pd.cut(df[CFG.target_col].values.reshape(-1), bins=num_bins, labels=False)
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["bins"], df["site_id"])):
        df.loc[val_idx, 'fold'] = fold
    return df