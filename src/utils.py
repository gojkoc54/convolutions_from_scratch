from cProfile import label
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os 
import numpy as np
import datetime

from models import *
from optim import BetaLASSO

# TODO:
#   - Add logging !

#   - Learning rate scheduler - Cosine Annealing as in the paper
#   - Augmentation - we won't use it


class Logger:
    def __init__(self, log_filepath=None):
        self.log_filepath = log_filepath
        if self.log_filepath is None:
            self.log_filepath = 'log.txt'

        dir_path = os.path.dirname(self.log_filepath)
        os.makedirs(dir_path, exist_ok=True)

    def log(self, msg, to_file=True, to_console=True):
        self.log_file = open(self.log_filepath, 'a')

        if to_file:
            self.log_file.write(f'{datetime.datetime.now()}\n')
            self.log_file.write(f'{msg}\n\n')
        if to_console:
            print(msg)

        self.log_file.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_true_parameters_count(model, alpha, img_size, num_classes):
    if model == 'D-CONV':
        cnt = (9 * 2**8 + 4 * img_size[0]**2) * alpha**2 
        cnt += (27 + 64 * num_classes) * alpha 
    
    elif model == 'S-CONV':
        cnt = 6 * img_size[0]**2 * alpha**2 + (243 + 24 * num_classes) * alpha
    
    elif model == 'D-FC':
        cnt = (img_size[0]**4 + 4 * img_size[0]**2) * alpha**2
        cnt += (3 * img_size[0]**4 + 64 * num_classes) * alpha

    elif model == 'S-FC':
        cnt = 6 * img_size[0]**2 * alpha**2 
        cnt += (3 * img_size[0]**4 / 4 + 24 * num_classes) * alpha

    elif model == 'D-LOCAL':
        cnt = 13 * img_size[0]**2 * alpha**2
        cnt += (27 * img_size[0]**2 + 64 * num_classes) * alpha

    elif model == 'S-LOCAL':
        cnt = 6 * img_size[0]**2 * alpha**2 
        cnt += (243 * img_size[0]**2 / 4 + 24 * num_classes) * alpha

    return cnt 


class MetricTracker:

    def __init__(self):
        self.batches_cnt = 0
        self.total_loss, self.avg_loss = 0, 0
        self.hits_cnt, self.samples_cnt = 0, 0

    def update(self, loss, preds, labels, paths=None):
        self.batches_cnt += 1
        
        self.total_loss += float(loss)
        self.avg_loss = self.total_loss / self.batches_cnt

        preds_classes = preds.argmax(dim=1)
        self.samples_cnt += labels.shape[0]
        self.hits_cnt += (preds_classes == labels).sum()

    def get_accuracy(self):
        return 100 * (self.hits_cnt / self.samples_cnt)


def initialize_model(model_name, model_params):
    if model_name == 's-fc':
        model = SFC(**model_params)
    elif model_name == 's-conv':
        model = SConv(**model_params)
    elif model_name == 's-local':
        model = SLocal(**model_params)

    return model


def initialize_optimizer(optimizer_name, model, optimizer_params):
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == 'beta-lasso':
        optimizer = BetaLASSO(model.parameters(), **optimizer_params)

    return optimizer


def save_checkpoint(model, save_path, val_loss=-1, epoch=-1):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss
        },
        save_path
        )

    print(f'\nCheckpoint saved to {save_path}\n')


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_idx=0):
    
    # Put the model in training mode
    model.train()
    
    # Initialize the metrics
    metric_tracker = MetricTracker()

    total_batch_num = len(dataloader.dataset) // dataloader.batch_size
    
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels_one_hot = torch.nn.functional.one_hot(labels).to(torch.float32) 
        
        # Reset the optimizer gradient
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(inputs)
        
        # Calculate the loss
        loss = criterion(preds, labels_one_hot)
        
        # Backwards pass and parameter update
        loss.backward()
        optimizer.step()
        
        # Update the metrics
        metric_tracker.update(float(loss), preds, labels)
        
        if i % 10 == 0:
            msg = f'Train epoch {epoch_idx},'
            msg += f' Batch {i}/{total_batch_num}: {float(loss)}'
            print(msg)
    
    print(f'\n=== TRAIN - Epoch {epoch_idx} ===')
    print(f'Avg loss = {metric_tracker.avg_loss}')
    print(f'Accuracy = {metric_tracker.get_accuracy()}')
    print()
    
    return metric_tracker


def evaluate(model, dataloader, criterion, device, title=None):
    
    if title is None:
        title = 'VALIDATION'

    # Put the model in eval mode
    model.eval()
    
    # Initialize the metrics
    metric_tracker = MetricTracker()

    total_batch_num = len(dataloader.dataset) // dataloader.batch_size
    
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels_one_hot = torch.nn.functional.one_hot(labels).to(torch.float32)
        
        # Forward pass
        preds = model(inputs)
            
        loss = criterion(preds, labels_one_hot)
        
        # Update the metrics
        metric_tracker.update(float(loss), preds, labels)
        
        if i % 10 == 0:
            msg = f'{title.capitalize()} - Batch '
            msg += f'{i}/{total_batch_num}: {float(loss)}'
            print(msg)
    
    print(f'\n=== {title} ===')
    print(f'Avg loss = {metric_tracker.avg_loss}')
    print(f'Accuracy = {metric_tracker.get_accuracy()}')
    print()
    
    return metric_tracker


def fit(model, loaders, optimizer, criterion, device, checkpoints_path, epochs=10):
    
    checkpoint_save_path = os.path.join(
        checkpoints_path, 
        f'{type(model).__name__.lower()}_checkpoint.pt'
        )

    metrics_save_path = os.path.join(
        checkpoints_path, 
        f'{type(model).__name__.lower()}_metrics_per_epoch.pt'
        )   

    train_loader, test_loader = loaders
    metrics_per_epoch = {}

    # Training loop (over epochs)
    for epoch in range(epochs):

        metrics_per_epoch[epoch] = {}
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
            )
        metrics_per_epoch[epoch]['train'] = train_metrics

        # Save the model checkpoint
        save_checkpoint(model, checkpoint_save_path, -1, epoch)

        # Save the metrics
        torch.save(metrics_per_epoch, metrics_save_path)

    # Testing on the test set
    model.load_state_dict(
        torch.load(checkpoint_save_path)['model_state_dict']
        )
    test_metrics = evaluate(model, test_loader, criterion, device, 'TESTING')
    metrics_per_epoch['test'] = test_metrics

    # Save the metrics
    torch.save(metrics_per_epoch, metrics_save_path)
    print(f'\nMetrics saved to {metrics_save_path}\n')

    return model


def load_datasets(dataset_str='cifar10', batch_size=256, num_workers=0):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    if dataset_str == 'cifar10':
        train_data = datasets.CIFAR10(
            '../data/cifar', train=True, transform=train_transform, download=True
            )
        test_data = datasets.CIFAR10(
            '../data/cifar', train=False, transform=test_transform, download=True
            )
        num_classes = 10
    elif dataset_str == 'cifar100':
        train_data = datasets.CIFAR100(
            '../data/cifar', train=True, transform=train_transform, download=True
            )
        test_data = datasets.CIFAR100(
            '../data/cifar', train=False, transform=test_transform, download=True
            )
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
        )

    return (train_loader, test_loader), num_classes

