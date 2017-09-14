#!/usr/bin/env bash
set -e

DATETIME=$(date "+%Y-%m-%d_%H_%M")
PLATFORM=$(uname)

PRJ_DIR=$(dirname $0)
mkdir -p $PRJ_DIR/var/log
LOG=$PRJ_DIR/var/log/$DATETIME.log

EMB_DIM=(32 64)
HIDDEN=(64 128)
NLAYER=(2 4)
DROPOUT=(0.1 0.3)
LR=(1.0 0.5 0.1)
BATCH_SIZE=64

if [ "$PLATFORM" == "Linux" ]; then
  CMD="stdbuf -o 0 python3 ./train.py --epochs 10 -b 512"
elif [ "$PLATFORM" == "Darwin" ]; then
  CMD="script -q /dev/null python3 ./train.py -b 64 "
else
  echo "Unsupported platform: $PLATFORM"
  exit 1
fi

for emb in "${EMB_DIM[@]}"; do
  for hidden in "${HIDDEN[@]}"; do
    for nlayer in "${NLAYER[@]}"; do
      for dropout in "${DROPOUT[@]}"; do
        echo "START: --embedding-size $emb --hidden-size $hidden --nlayers $nlayer --dropout $dropout" | tee -a $LOG
        date | tee -a $LOG
        $CMD --embedding-size $emb --hidden-size $hidden --nlayers $nlayer --dropout $dropout | tee -a $LOG 2>&1
        date | tee -a $LOG
        echo "END: --embedding-size $emb --hidden-size $hidden --nlayers $nlayer --dropout $dropout" | tee -a $LOG
      done
    done
  done
done
