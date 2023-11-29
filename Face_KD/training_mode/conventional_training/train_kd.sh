# mkdir 'log'
# --step '10, 13, 16' \
python3 train_kd.py \
    --data_root '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/train.pkl' \
    --test_data '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test.pkl' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.01 \
    --epoches 25 \
    --step '5, 10, 15'\
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 64 \
    --momentum 0.9 \
    --log_dir 'log' \
    --out_dir 'out_dir' \
    --finetune \
    --temperature 2 \
    --alpha 1.0 \
    --teacher 'r50' \
    --tensorboardx_logdir 'logits_af_anneal2_lr0.01_r50S_r50T_KD' \
    2>&1 | tee log/logits_af_anneal2_lr0.01_r50S_r50T_KD.log
    &> train_kd.out
