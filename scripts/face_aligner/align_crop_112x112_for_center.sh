#/usr/bin/evn bash
num_gpu=1
# declare -x GLOG_minloglevel=2

echo 'num_gpu: ' $num_gpu

nohup python -u batch_mtcnn_align_crop_112x112_for_center.py \
        --rect-root-dir=/workspace/data/qyc/data/MeGlass_ori_Crop \
        --mtcnn-model-dir=../../model \
        --save-dir=/workspace/data/qyc/data/MeGlass_ori_align \
        --gpu-id=0 > ./process-log.txt &

