python train.py --prefix vit_cn7a_D100 --from_check_point "model_vit_cn7aD100.pth" --size 240 --patch 10 --dataset ../imagenet --batchsize 64 --cvd deutan --tau 1.0 --train_mode optim

# python train.py --prefix vit_cn7a --size 240 --patch 10 --dataset ../imagenet --batchsize 64 --cvd deutan --train_mode est