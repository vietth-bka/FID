# mkdir 'log'
CUDA_VISIBLE_DEVICES=2 python train_dml_seq.py \
    --data_root '/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/train.pkl' \
    --test_data '/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test.pkl' \
    --head_type 'ArcFace' \
    --backbone_type 'HRNet' \
    --backbone_conf_file '../backbone_conf.yaml'\
    --head_conf_file '../head_conf.yaml' \
    --lr 0.001 \
    --epoches 50 \
    --step '5, 10, 15'\
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 64 \
    --momentum 0.9 \
    --log_dir 'log' \
    --out_dir 'out_dir' \
    --finetune \
    --temperature 6 \
    --alpha 1.0 \
    --teacher 'r100' \
    --tensorboardx_logdir 'dml_af_lr1e-3_HRnetS_r100T_seq32' \
    2>&1 | tee log/dml_af_lr1e-3_HRnetS_r100T_seq32.log