TOTAL_NUM_UPDATES=5000 
WARMUP_UPDATES=200      
LR=3e-05
MAX_TOKENS=800
UPDATE_FREQ=32
BART_PATH=PATH-TO-BART-MODEL (./bart.large/model.pt)

CUDA_VISIBLE_DEVICES=0 python train.py cnn_dm-bin_2 \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --required-batch-size-multiple 1 \
    --no-epoch-checkpoints \
    --save-dir checkpoints\
    --lr-weight 1000 \
    --T 0.2 \
    --multi-views \
    --balance \
    --seed 14632











