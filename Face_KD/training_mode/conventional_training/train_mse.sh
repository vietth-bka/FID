mkdir 'log'
python3 train_mse.py \
    --data_root '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/train.pkl' \
    --test_data '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test.pkl' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 1e-3 \
    --out_dir 'out_dir' \
    --epoches 25 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 64 \
    --momentum 0.9 \
    --log_dir 'log' \
    --finetune \
    --lambdaa 1000 \
    --teacher 'r100' \
    --tensorboardx_logdir 'mse_af_lb1000_r50S_r100T_KD' \
    2>&1 | tee log/mse_af_lb1000_r50S_r100T_KD.log

# python3 train.py \
#     --data_root '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/train.pkl' \
#     --test_data '/media/v100/DATA2/vietth/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test.pkl' \
#     --head_type 'ArcFace' \
#     --head_conf_file '../head_conf.yaml' \
#     --lr 0.1 \
#     --out_dir 'out_dir' \
#     --epoches 25 \
#     --alpha 1. \
#     --temperature 5. \
#     --step '10, 13, 16' \
#     --print_freq 200 \
#     --save_freq 3000 \
#     --batch_size 64 \
#     --momentum 0.9 \
#     --log_dir 'log' \
#     --finetune \
#     --tensorboardx_logdir 'af_1._r50_KD' \
#     2>&1 | tee log/af_1._r50_KD.log