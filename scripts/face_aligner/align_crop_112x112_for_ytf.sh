#/usr/bin/evn bash
num_gpu=1
# declare -x GLOG_minloglevel=2

echo 'num_gpu: ' $num_gpu

nohup python batch_mtcnn_align_crop_112x112_for_ytf.py \
        --image-root-dir=/workspace/data/youtubefaces/YoutubeFacesDB/frame_images_DB \
        --rect-root-dir=/workspace/data/youtubefaces/YoutubeFacesDB/frame_images_DB \
        --mtcnn-model-dir=../../model \
        --save-dir=/workspace/data/youtubefaces/align \
        --gpu-id=0 > ./process-log.txt &

