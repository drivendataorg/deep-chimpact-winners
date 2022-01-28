import pandas as pd
import os
from dataset.processing import video2image
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/raw', help='directory of raw data')
    parser.add_argument('--save-dir', type=str, default='data/processed', help='directory of processed data')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--infer-only',action='store_true',help='generate only the test files')
    opt = parser.parse_args()
    
    DATA_DIR = opt.data_dir
    SAVE_DIR = opt.save_dir
    DEBUG    = opt.debug
    INFER_ONLY = opt.infer_only
    
    # Read Meta Data
    train_df               = pd.read_csv(f'{DATA_DIR}/train_metadata.csv')
    train_labels           = pd.read_csv(f'{DATA_DIR}/train_labels.csv')
    train_df               = train_df.merge(train_labels, on=['video_id','time'],how='left')
    train_df['video_path'] = f'{DATA_DIR}/train_videos/'+train_df.video_id

    test_df                = pd.read_csv(f'{DATA_DIR}/test_metadata.csv')
    test_df['video_path']  = f'{DATA_DIR}/test_videos/'+test_df.video_id
    
    sub_df                 = pd.read_csv(f'{DATA_DIR}/submission_format.csv')
    
    # Create Image Directory
    os.makedirs(f'{SAVE_DIR}/train_images', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/test_images', exist_ok=True)
    
    # Train
    if not INFER_ONLY:
        print('Train:')
        info     = video2image(train_df, image_dir=f'{SAVE_DIR}/train_images', debug=DEBUG)
        info_df  = pd.DataFrame(info, columns=['video_id', 'width', 'height'])
        train_df = train_df.merge(info_df, on='video_id', how='left')
    
    # Test
    print('\nTest:')
    info    = video2image(test_df, image_dir=f'{SAVE_DIR}/test_images', debug=DEBUG)
    info_df = pd.DataFrame(info, columns=['video_id', 'width', 'height'])
    test_df = test_df.merge(info_df, on='video_id', how='left')
    
    # Meta-Data
    train_df.to_csv(f'{SAVE_DIR}/train.csv',index=False)
    test_df.to_csv(f'{SAVE_DIR}/test.csv',index=False)
    sub_df.to_csv(f'{SAVE_DIR}/sample_submission.csv',index=False)
    
    