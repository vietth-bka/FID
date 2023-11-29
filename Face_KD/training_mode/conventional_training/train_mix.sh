# mkdir 'log'
python train_mix.py \
    --data_root '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/train.pkl' \
    --test_data '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test.pkl' \
    --head_type 'ArcFace' \
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
    --temperature 2 \
    --alpha 1.0 \
    --teacher 'r100' \
    --tensorboardx_logdir 'mix_af_anneal_lr0.01_r100S_r100T_layer1_noAdapt_KD' \
    2>&1 | tee log/mix_af_anneal_lr0.01_r100S_r100T_layer1_noAdapt_KD.log