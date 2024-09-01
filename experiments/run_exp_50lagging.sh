#!/bin/bash
target_dir=tmp_result_50lagging/logs
mkdir -p $target_dir
for ins_num in {1..5}; do
    python ../03_run_target_lagging.py --lr 0.01 --problem_size sm --ins_num ${ins_num} --gamma 0.99 > $target_dir/lr0.01_gamma0.99_sm${ins_num}_clip0.8.log
done
