#/usr/bin/evn bash
# declare -x GLOG_minloglevel=2


nohup python -u detect_image_list.py \
      /workspace/data/qyc/data/MeGlass_ori_classify.lst \
      /workspace/data/qyc/data/MeGlass_ori_classify \
      /workspace/data/qyc/data/MeGlass_ori_Crop \
     > ./nohup.log 2>&1 &

