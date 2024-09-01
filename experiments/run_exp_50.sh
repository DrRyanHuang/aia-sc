#!/bin/bash
target_dir=tmp_result_50/logs
mkdir -p $target_dir
for ins_num in {11..20}; do
    python ../02_run.py --lr 0.01 --problem_size sm --ins_num ${ins_num} --gamma 0.99 > $target_dir/lr0.01_gamma0.99_sm${ins_num}_clip0.8.log
done
