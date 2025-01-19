#!/usr/bin/env bash
set -x
DATAPATH="/data/stereo/sceneflow"
CUDA_VISIBLE_DEVICES="0,1"  python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 16 --lrepochs "10,12,14,16:2" --batch_size 4 --test_batch_size 2 \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow\
    --loadckpt ./pretrained/sceneflow/gwcnet-gc/checkpoint_000015.ckpt