#!/bin/bash
target_dir=tmp_result_50simpleloss/logs
mkdir -p $target_dir
for ins_num in {6..10}; do
    python ../04_run_simpleloss.py --lr 0.01 --problem_size sm --ins_num ${ins_num} --gamma 0.99 > $target_dir/lr0.01_gamma0.99_sm${ins_num}_clip0.8.log
done
