#!/bin/bash
ulimit -n 64000

rm -f ./train.log
python ./multilabel_train_enhanced.py $@
#python ./multilabel_train_enhanced.py $*


