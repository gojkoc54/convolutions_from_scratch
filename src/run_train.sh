models=( "s-fc" )
beta_array=( 0 1 50 )
EPOCHS=400
LR=0.1
LAMBDA="2e-5"
DATASET="svhn"

for model in "${models[@]}"
do
    # Basic SGD
    python train.py --model ${model} --dataset $DATASET --optimizer beta-lasso \
        --epochs $EPOCHS --lr $LR --beta 0 --lambda_ 0 

    # Beta-LASSO
	for beta in "${beta_array[@]}"
    do
        python train.py --model ${model} --dataset $DATASET --optimizer beta-lasso \
            --epochs $EPOCHS --lr $LR --beta $beta --lambda_ $LAMBDA
    done
done


