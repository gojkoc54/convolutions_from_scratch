import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore")
import os
import argparse

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--lambda_', default=1e-1, type=float)
parser.add_argument('--bs', '--batch-size', default=512, type=int)
parser.add_argument('--alpha', '--base-channels', default=1, type=int)
parser.add_argument('--img-size', default=(32, 32), type=tuple)

parser.add_argument('--model', default='s-fc', type=str)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--root', default='/workspace/convs_from_scratch', type=str)
parser.add_argument('--cp-path', default='checkpoints', type=str)
parser.add_argument('--log-path', default='../logs/training_log.txt', type=str)

args = parser.parse_args()


if __name__ == '__main__':

    # Define the destination device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice being used: {device}\n')
    
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Loading the Data Loaders
    (train_loader, test_loader), num_classes = \
        load_datasets(args.dataset, args.bs)
    loaders = [train_loader, test_loader]

    # Initialize the model
    model_params = {
        'in_channels': 3, 'base_channels': args.alpha,
        'img_size': args.img_size, 'num_classes': num_classes
        }

    model = initialize_model(args.model, model_params)
    print(f'\nInitialized model {type(model).__name__}\n')

    # If the model was properly loaded, create the checkpoint directory
    checkpoint_name = f'{args.model}_beta_{args.beta}_lr_{args.lr}'
    checkpoint_name += f'{checkpoint_name}_scheduler_lambda_{args.lambda_}'
    checkpoints_path = os.path.join(args.root, args.cp_path, checkpoint_name)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Move model to GPU
    model = model.to(device)

    # Define the optimizer
    # optimizer_params = {'lr': args.lr}
    optimizer_params = {
        'lr': args.lr, 'beta': args.beta, 'lambda_': args.lambda_
        }

    optimizer = initialize_optimizer(args.optimizer, model, optimizer_params)
    print(f'Initialized optimizer {type(optimizer).__name__}\n')
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)
    
    # Start training  
    model = fit(
        model, loaders, optimizer, lr_scheduler, criterion, 
        device, checkpoints_path, args.epochs
        )

