#!/bin/bash
dev=2
setting=workplace/log_5000_machine_annotation
#setting=workplace/dual_adapter_baseline_bsz10

model_path=$setting/wiki.dailydiag_bleu5
test_path=../data/5000_machine_annotation/test
#test_path=../data/daily_semi/dev
ref_path=../data/5000_machine_annotation/test.tgt
#ref_path=../data/daily_semi/dev.tgt
out_file=$setting/eval.log
#out_file=$setting/eval-dev.log
#step=30


for((step=10;step<=100;step+=10))
do
hyp_path=$setting/test-$step
CUDA_VISIBLE_DEVICES=$dev python decode-fast.py --prefix_path $model_path --in_path $test_path --out_path $hyp_path --checkpoint_step $step --beam_size 5 --batch_size 40
echo "================step${step}=============" >> $out_file
python evaluation/eval_v2.py ${hyp_path}.hyp $ref_path | tee -a $out_file
#perl evaluation/multi-bleu.perl $ref_path < ${hyp_path}.hyp | tee -a $out_file
done



#for((step=200;step<=200;step+=10))
#do
#hyp_path=$setting/test-$step
##hyp_path=$setting/dev-$step
#CUDA_VISIBLE_DEVICES=$dev python decode-fast.py --prefix_path $model_path --in_path $test_path --out_path $hyp_path --checkpoint_step $step --beam_size 5 --batch_size 40
#echo "================step${step}=============" >> $out_file
#python evaluation/eval_v2.py ${hyp_path}.hyp $ref_path | tee -a $out_file
#perl evaluation/multi-bleu.perl $ref_path < ${hyp_path}.hyp | tee -a $out_file
#done