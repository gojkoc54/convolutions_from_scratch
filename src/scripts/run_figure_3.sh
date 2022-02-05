MODEL="s-fc"
BETA=50
EPOCHS=100
LR=0.1
LAMBDA="2e-5"

# Run the training script for S-FC, with CIFAR-10
python train.py --model ${MODEL} --dataset cifar10 --optimizer beta-lasso \
        --epochs $EPOCHS --lr $LR --beta $BETA --lambda_ $LAMBDA --cp-path checkpoints/cifar10_${BETA}

# Run the training script for S-FC, with SVHN
python train.py --model ${MODEL} --dataset svhn --optimizer beta-lasso \
        --epochs $EPOCHS --lr $LR --beta $BETA --lambda_ $LAMBDA --cp-path checkpoints/svhn_${BETA}

