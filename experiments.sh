#!/usr/bin/env bash
set -e

DATETIME=$(date "+%Y-%m-%d_%H_%M")
echo $DATETIME

PRJ_DIR=$(dirname $0)
mkdir -p $PRJ_DIR/var/log
LOG=$PRJ_DIR/var/log/$DATETIME.log

EMB_DIM=(32 64)
HIDDEN=(64 128)
NLAYER=(2 4)
DROPOUT=(0.1 0.3)
LR=(1.0 0.5 0.1)
BATCH_SIZE=64

for emb in "${EMB_DIM[@]}"; do
  for hidden in "${HIDDEN[@]}"; do
    for nlayer in "${NLAYER[@]}"; do
      for dropout in "${DROPOUT[@]}"; do
        echo "START: -b 64 --embedding-size $emb --hidden-size $hidden --nlayers $nlayer --dropout $dropout" \
          | tee -a $LOG
        date | tee -a $LOG
        script -q /dev/null ./train.py -b 64 --embedding-size $emb --hidden-size $hidden --nlayers $nlayer \
          --dropout $dropout | tee -a $LOG 2>&1
        date | tee -a $LOG
        echo "END: -b 64 --embedding-size $emb --hidden-size $hidden --nlayers $nlayer --dropout $dropout" | tee -a $LOG
      done
    done
  done
done
