PROJ_PATH=/home/scratch.sysarch_nvresearch/chaowei/esvit
DATA_PATH=/home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012/

OUT_PATH=$PROJ_PATH/exp_output
CKPT_PATH=$PROJ_PATH/model_checkpoints/$1/checkpoint.pth
ARCH=$2

# python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 1000 MODEL.NUM_CLASSES 0

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 eval_knn_c.py --data_path $DATA_PATH \
                            --dump_features $OUT_PATH/$ARCH/features --pretrained_weights $CKPT_PATH \
                            --log_dir $OUT_PATH/$ARCH/knn_c
                            --checkpoint_key teacher --batch_size_per_gpu 256 --arch $ARCH \
                            --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml MODEL.NUM_CLASSES 0\
                            --load_features exp_output/swin_tiny/features