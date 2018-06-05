#/usr/bin/evn bash
num_gpu=1
# declare -x GLOG_minloglevel=2

echo 'num_gpu: ' $num_gpu

nohup python batch_mtcnn_align_crop_112x112_for_bh.py \
        --rect-root-dir=../../data/bh_rlt4/ \
        --mtcnn-model-dir=../../model \
        --save-dir=../../data/ \
        --gpu-id=0 > ./process-log.txt &

