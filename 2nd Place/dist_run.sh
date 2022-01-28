#!/usr/bin/env sh


set -eu


GPU=${GPU:-0,1,2,3,4}
PORT=${PORT:-29500}
N_GPUS=5

IN_CHANNELS=3

OPTIM=fusedadam
LR=0.001
WD=0.000005

SCHEDULER=cosa
T_MAX=25
MODE=epoch

N_EPOCHS=80

MODEL_NAME=tf_efficientnetv2_l_in21k
WIN_SIZE=2
for FOLD in {0..9}; do
    CHECKPOINT=./chkps/"${MODEL_NAME}"_"${WIN_SIZE}"_"${FOLD}"_TEST
    LOAD="${CHECKPOINT}"/model_last.pth

    CUDA_VISIBLE_DEVICES="${GPU}" python3 -m torch.distributed.launch --nproc_per_node="${N_GPUS}" --master_port="${PORT}" \
        ./dist_train.py \
            --train-df ./data/train.csv \
            --train-images-dir ./data/train_images \
            --encoder-name "${MODEL_NAME}" \
            --in-channels "${IN_CHANNELS}" \
            --optim "${OPTIM}" \
            --learning-rate "${LR}" \
            --weight-decay "${WD}" \
            --scheduler "${SCHEDULER}" \
            --T-max "${T_MAX}" \
            --num-epochs "${N_EPOCHS}" \
            --checkpoint-dir "${CHECKPOINT}" \
            --distributed \
            --fp16 \
            --fold "${FOLD}" \
            --window-size "${WIN_SIZE}" \
            --load "${LOAD}" \
            --resume \
            --scheduler-mode "${MODE}" \

done
