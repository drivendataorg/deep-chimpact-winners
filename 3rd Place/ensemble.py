#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
parser.add_argument('--data_preds_dir', type=str, default='preds', help='Prediction directory')
parser.add_argument('--out_dir', type=str, default='./', help='Output directory')
parser.add_argument('--out_name', type=str, default='submission.csv', help='Submission file name')
parser.add_argument('--n_folds', type=int, default=5, help='N folds')
parser.add_argument('--n_tta', type=int, default=0, help='N TTA')
parser.add_argument('--compute_weights', type=str, default='False', choices=['True', 'False'], 
                    help='Whether to optimize ensemble weights (False: use precomputed weights, True: OOF predictions are required)')
args = parser.parse_args()

import os
import sys
sys.path.append('lib')
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from vecxoz_utils import create_cv_split

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

os.makedirs(args.out_dir, exist_ok=True)

# Collect model dirs to use for ensemble
dirs = sorted(glob.glob(os.path.join(args.model_dir, '*')))

train_df, subm_df = create_cv_split(args.data_dir, args.n_folds)

def nearest(x):
    """Returns nearest to x value from the list"""
    return min(unique_labels_sorted, key=lambda u: abs(u - x))
closest_vect = np.vectorize(nearest)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Ensembling %d models...' % len(dirs))

if args.compute_weights == 'True':

    print('Collecting VAL predictions...')
    y_preds = []
    for counter, d in enumerate(dirs):
        all_tta = []
        for tta_id in range((args.n_tta + 1)):
            all_folds = []
            for fold_id in range(args.n_folds):
                all_folds.append(np.load(os.path.join(d, args.data_preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel())
            all_tta.append(np.hstack(all_folds))
        y_preds.extend(all_tta)
        print(counter, end='\r')
    assert len(y_preds) == (args.n_tta + 1) * len(dirs)
    
    # Collect coresponding true label
    y_true_list = []
    for fold_id in range(args.n_folds):
        y_true_list.append(train_df.loc[train_df['fold_id'] == fold_id, 'distance'].values.ravel())
    y_true = np.hstack(y_true_list)
    
    # Compute scores
    scores = []
    for y_pred in y_preds:
        score = mean_absolute_error(y_true, y_pred)
        scores.append(score)
    
    # Sort based on scores, ascending (best scores first), each prediction in a row
    sorting_ids = np.argsort(scores)
    y_preds_sorted = np.array(y_preds)[sorting_ids]
    
    #----
    
    print('Optimizing ensemble coefs...')
    
    # Range of coefs
    coef_range = np.arange(0, 1.11, 0.01)
    
    coefs_best = []
    pred_best = y_preds_sorted[0]
    score_best = 100
    
    for pred_id in range(1, (args.n_tta + 1) * len(dirs)):
        coef_best = 1
        for coef in coef_range:
            y_pred = coef * pred_best + (1 - coef) * y_preds_sorted[pred_id]
            score_current = mean_absolute_error(y_true, y_pred)
            if score_current < score_best:
                score_best = score_current
                coef_best = coef
        coefs_best.append(coef_best)
        pred_best = coef_best * pred_best + (1 - coef_best) * y_preds_sorted[pred_id]
    
    assert len(coefs_best) == (args.n_tta + 1) * len(dirs) - 1
    
    # CHECK: apply best coefs to VAL
    y_pred_final = y_preds_sorted[0]
    for pred_id in range(1, (args.n_tta + 1) * len(dirs)):
        y_pred_final = coefs_best[pred_id-1] * y_pred_final + (1 - coefs_best[pred_id-1]) * y_preds_sorted[pred_id]
    print('BEST score: %.6f' % mean_absolute_error(y_true, y_pred_final))
    
    # Shift to the nearest value present in training data
    unique_labels_sorted = np.array(sorted(np.unique(y_true)))
    print('BEST score (nearest): %.6f' % mean_absolute_error(y_true, closest_vect(y_pred_final)))

elif args.compute_weights == 'False':

    sorting_ids = np.array([10,  9, 11,  5,  8,  7,  2,  0,  6,  1,  3,  4])
    coefs_best = np.array([0.58, 0.72, 0.83, 0.81, 0.89, 0.87, 0.98, 0.88, 1.03, 1.07, 1.1])
    unique_labels_sorted = np.array([ 
        0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,
        5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  9. , 10. , 11. , 12. , 13. ,
       13.5, 14. , 15. , 16. , 17. , 18. , 19. , 20. , 21. , 22. , 23. ,
       24. , 25. ])

#------------------------------------------------------------------------------
# Create submission for test set
#------------------------------------------------------------------------------

print('Collecting TEST predictions...')
y_preds_test = []
for counter, d in enumerate(dirs):
    for tta_id in range((args.n_tta + 1)):
        y_preds_folds = []
        for fold_id in range(args.n_folds):
            y_preds_folds.append( np.load(os.path.join(d, args.data_preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel() )
        y_preds_test.append(np.mean(y_preds_folds, axis=0))
    print(counter, end='\r')
assert len(y_preds_test) == (args.n_tta + 1) * len(dirs)

# Sort according to val predictions order, each prediction in a row
y_preds_test_sorted = np.array(y_preds_test)[sorting_ids]

# Apply best coefs
y_pred_test_final = y_preds_test_sorted[0]
for pred_id in range(1, (args.n_tta + 1) * len(dirs)):
    y_pred_test_final = coefs_best[pred_id-1] * y_pred_test_final + (1 - coefs_best[pred_id-1]) * y_preds_test_sorted[pred_id]

print('Creating submission...')
subm_df['distance'] = closest_vect(y_pred_test_final)
subm_df[['video_id', 'time', 'distance']].to_csv(os.path.join(args.out_dir, args.out_name), index=False)
subm_df[['video_id', 'time', 'distance']].head()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

