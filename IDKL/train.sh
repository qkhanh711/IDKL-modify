tmux new-session -d -s session1 "python train.py --cfg ./configs/RegDB.yml --base_dim 2048 --model_name resnet50"
tmux new-session -d -s session2 "python train.py --cfg ./configs/RegDB.yml --base_dim 320 --model_name efficientnet"
tmux new-session -d -s session3 "python train.py --cfg ./configs/RegDB.yml --base_dim 192 --model_name shufflenetv2"
tmux new-session -d -s session4 "python train.py --cfg ./configs/RegDB.yml --base_dim 16 --model_name mnasnet"