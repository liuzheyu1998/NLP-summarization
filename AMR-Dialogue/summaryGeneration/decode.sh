setting=workplace/dual_baseline_debug3
#setting=workplace/dual_adapter_baseline_bsz10

model_path=$setting/wiki.dailydiag_bleu5
test_path=../data/daily_semi/test
#test_path=../data/daily_semi/dev
#ref_path=../data/daily_semi/test.tgt
#ref_path=../data/daily_semi/dev.tgt
out_file=$setting/eval.log
step=180

CUDA_VISIBLE_DEVICES=1 python decode.py --prefix_path $model_path \
    --in_path test_path --out_path $out_file --checkpoint_step $step --beam_size 5
