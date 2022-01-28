from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.metrics import mean_absolute_error
from utils.submission import get_shortest_distance
from utils.pbar import tqdm_bar

def show_msg(txt='hello world'):
    print('='*5+f' {txt} '+'='*5)

def sort_list(paths):
    ## DESCENDING ORDER
    #try:
    sorted_paths= sorted(paths,key=lambda x:int(os.path.basename(x).split('x')[-1].split('.')[0]),reverse=True)
    #except:
       # print('Size wise sorting failed, returning default sort')
       # sorted_paths=sorted(paths,reverse=True)
    return sorted_paths

    
class MeanEnsemble(object):
    def __init__(self, weights=None,indices=None,sort=True):
        self.weights=weights
        self.indices=indices
        self.sort= True
        

    def get_paths(self, output_dir):
        
        def get_paths_from_dir(output_dir):
            oof_paths = glob(output_dir+'/*/oof.csv',recursive=True)
            sub_paths = glob(output_dir+'/*/submission.csv',recursive=True)
            if len(oof_paths)+len(sub_paths)==0:
                new_oof_paths=glob(output_dir+'/*oof.csv',recursive=True)
                new_sub_paths=glob(output_dir+'/*submission.csv',recursive=True)
                if len(new_oof_paths)+len(new_sub_paths)>0:
                    return new_oof_paths,new_sub_paths
            return oof_paths, sub_paths
        
        oof_paths, sub_paths=[],[]
        if isinstance(output_dir,list):
            for dir_name in output_dir:
                oofs,subs=get_paths_from_dir(dir_name)
                oof_paths.extend(oofs)
                sub_paths.extend(subs)
        else:
            oof_paths, sub_paths   = get_paths_from_dir(output_dir)
            
        if self.sort:
            oof_paths, sub_paths = sort_list(oof_paths), sort_list(sub_paths)
        return oof_paths, sub_paths

    def load_dfs(self, oof_paths, desc=''):
        dfs = [pd.read_csv(path).sort_values(by=['video_id','time']) for path in tqdm(oof_paths, 
                                                                                      desc=f'{desc} ')]
        return dfs
    
    def get_data_sub(self, dfs):
        preds = np.zeros((len(dfs[0]),len(dfs)))
        try:
            for idx in range(len(dfs)):
                preds[:,idx] = dfs[idx].pred.values
            true = dfs[0].true.values
            return true, preds
        except:
            for idx in range(len(dfs)):
                preds[:,idx] = dfs[idx].distance.values
            return preds           
        
    def get_data_oof(self, dfs):
        preds = np.zeros((len(dfs[0]),len(dfs)))
        for idx in range(len(dfs)):
            preds[:,idx] = dfs[idx].pred.values
        try:
            true = dfs[0].true.values
        except:
            true = dfs[0].distance.values
        return true, preds
   
    
    def get_scores(self, true, preds):
        scores = []
        for idx in range(preds.shape[1]):
            pred = preds[:,idx]
            mae   = mean_absolute_error(true, pred)
            scores.append(mae)
        return scores
    
    def fit(self, output_dir):
    
        self.oof_paths, self.sub_paths   = self.get_paths(output_dir)
        show_msg('Loading CSV')
        self.oof_dfs          = self.load_dfs(self.oof_paths,'oof')
        self.sub_dfs          = self.load_dfs(self.sub_paths, 'sub')
        self.true, self.oof_preds  = self.get_data_oof(self.oof_dfs)
        self.sub_preds        = self.get_data_sub(self.sub_dfs)
        scores                = self.get_scores(self.true, self.oof_preds)
        print(); show_msg('Initial Result')
        for idx in range(len(scores)):
            print('Model %s has OOF MAE = %.5f'%(self.oof_paths[idx].split('/')[-2],scores[idx]))
        print(); print('='*5+' Optimize '+'='*5)    
        old          = np.min(scores); 

        self.indices= np.arange(len(self.oof_dfs)) if self.indices is None else self.indices
        self.weights = np.ones_like(self.oof_preds[0]) if self.weights is None else self.weights

        print('Initial  MAE = %.5f | Model = %i'%(old, self.indices[0]))


        print('Models                   =', self.indices)
        print('Weights                  =',self.weights)

        print()
        return 

    def fit_without_oof(self,output_dir,paths=None):
        if paths is None:
            _,self.sub_paths   = self.get_paths(output_dir)
        else:
            self.sub_paths = sort_list(paths) if self.sort else paths
        show_msg('Loading CSV')
        self.sub_dfs          = self.load_dfs(self.sub_paths, 'sub')
        self.sub_preds        = self.get_data_sub(self.sub_dfs)
        print('Models                   =', self.indices)
        print()
        return 
    
    def transform(self, save_dir='', rounding=False,with_oof=True):
        os.makedirs(save_dir, exist_ok=True) if len(save_dir)>0 else None
        show_msg('Ensemble')
        # ENSEMBLE OOF
        if with_oof:
            best_oof = self.oof_preds[:,self.indices[0]]
            for w_idx, m_idx in tqdm(enumerate(self.indices[1:]),
                                    desc='oof ',
                                    total=len(self.indices[1:]),
                                    bar_format=tqdm_bar):
                best_oof = self.weights[w_idx]*self.oof_preds[:,m_idx] + (1-self.weights[w_idx])*best_oof
            oof_df = self.oof_dfs[0].copy()
            oof_df.pred = best_oof
            mae = mean_absolute_error(self.true, best_oof)
            if rounding:
                tqdm.pandas(desc='rounding ', bar_format=tqdm_bar)
                oof_df = oof_df.progress_apply(get_shortest_distance, axis=1)
            oof_df.to_csv(os.path.join(save_dir,'ensemble_oof.csv'),index=False) if len(save_dir)>0 else None
            print()

        # ENSEMBLE SUBMISSION
        best_sub = self.sub_preds[:,self.indices[0]]
        for w_idx, m_idx in tqdm(enumerate(self.indices[1:]),
                                 desc='sub ',
                                 total=len(self.indices[1:]),
                                 bar_format=tqdm_bar):
            best_sub = self.weights[w_idx]*self.sub_preds[:,m_idx] + (1-self.weights[w_idx])*best_sub
        sub_df = self.sub_dfs[0].copy()
        sub_df.distance = best_sub
        if rounding:
            tqdm.pandas(desc='rounding ', bar_format=tqdm_bar)
            sub_df = sub_df.progress_apply(get_shortest_distance, axis=1)
        save_path = os.path.abspath(os.path.join(save_dir, 'ensemble_submission.csv'))
        sub_df.to_csv(save_path,index=False) if len(save_dir)>0 else None
        print(f'>> ENSEMBLED SUBMISSION SAVED TO {save_path}')
        if with_oof:
            print(); print('Ensemble MAE             = %.4f'%mae)
            show_msg('Done'); print()
            return oof_df, sub_df
        return sub_df
    
    def fit_transform(self, output_dir, rounding=False, save_dir='',with_oof=True,paths=None):
        if with_oof:
            self.fit(output_dir)
        else:
            self.fit_without_oof(output_dir,paths=paths)
        return self.transform(save_dir, rounding = rounding, with_oof = with_oof)


if __name__ == '__main__':

    index = [1, 3, 0, 4, 2, 0]
    weights = [1, 0.43, 0.34, 0.18, 0.125, 0.065]
    ens=MeanEnsemble(indices=index,weights=weights,sort=True)
    ens.fit_transform('checkpoints', rounding=True, save_dir='',with_oof=True)