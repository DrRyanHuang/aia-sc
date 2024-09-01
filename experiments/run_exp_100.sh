#!/bin/bash
target_dir=tmp_result_100/logs
mkdir -p $target_dir
for ins_num in {6..10}; do
    python ../02_run.py --lr 0.01 --problem_size m --ins_num ${ins_num} --gamma 0.99 > $target_dir/lr0.01_gamma0.99_m${ins_num}_clip0.8.log
done
