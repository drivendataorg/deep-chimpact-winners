#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------

DATA=$HOME/solution/data

mkdir -p $DATA
cd $DATA

aws s3 cp s3://drivendata-competition-depth-estimation-public-eu/test_videos_downsampled/ test_videos_downsampled/ --no-sign-request --recursive
aws s3 cp s3://drivendata-competition-depth-estimation-public-eu/test_videos/ test_videos/ --no-sign-request --recursive

# The following files are included in distribution (links are dynamic)
# submission_format.csv
# train_labels.csv

#-------------------------------------------------------------------------------
# Extract downsampled images
#-------------------------------------------------------------------------------

cd $HOME/solution

python3 extract_images_downsampled.py \
    --in_dir=$DATA/test_videos_downsampled \
    --out_dir=$DATA/test_images_downsampled \

#-------------------------------------------------------------------------------
# Select white downsampled images
#-------------------------------------------------------------------------------

cd $HOME/solution

python3 select_white_images_downsampled.py \
    --data_dir=$DATA \
    --out_dir=$DATA/test_images_downsampled_white \
    --subset=test \

#-------------------------------------------------------------------------------
# Extract full images with gamma correction for videos corresponding to the "white" images
#-------------------------------------------------------------------------------

cd $HOME/solution

python3 extract_images_full_with_gamma_correction.py \
    --in_dir=$DATA/test_videos \
    --out_dir=$DATA/test_images_full_gamma \
    --white_dir=$DATA/test_images_downsampled_white \

#-------------------------------------------------------------------------------
# Create TFRecords from downsampled images
#-------------------------------------------------------------------------------

cd $HOME/solution

# test, downsampled, step-1
python3 create_tfrecords.py \
    --data_dir=$DATA \
    --image_dir=$DATA/test_images_downsampled \
    --numpy_dir=$DATA/test_numpy_from_images_downsampled_step_1 \
    --tfrec_dir=$DATA/tfrec_from_images_downsampled_step_1 \
    --step=1 \
    --subset=test \
    --delete_numpy=True \

# test, downsampled, step-2
python3 create_tfrecords.py \
    --data_dir=$DATA \
    --image_dir=$DATA/test_images_downsampled \
    --numpy_dir=$DATA/test_numpy_from_images_downsampled_step_2 \
    --tfrec_dir=$DATA/tfrec_from_images_downsampled_step_2 \
    --step=2 \
    --subset=test \
    --delete_numpy=True \

#-------------------------------------------------------------------------------
# Create TFRecords from full images
#-------------------------------------------------------------------------------

cd $HOME/solution

# test, downsampled, step-1
python3 create_tfrecords.py \
    --data_dir=$DATA \
    --image_dir=$DATA/test_images_full_gamma \
    --numpy_dir=$DATA/test_numpy_from_images_full_gamma_step_1 \
    --tfrec_dir=$DATA/tfrec_from_images_full_gamma_step_1 \
    --step=1 \
    --subset=test \
    --delete_numpy=True \

# test, downsampled, step-2
python3 create_tfrecords.py \
    --data_dir=$DATA \
    --image_dir=$DATA/test_images_full_gamma \
    --numpy_dir=$DATA/test_numpy_from_images_full_gamma_step_2 \
    --tfrec_dir=$DATA/tfrec_from_images_full_gamma_step_2 \
    --step=2 \
    --subset=test \
    --delete_numpy=True \

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

