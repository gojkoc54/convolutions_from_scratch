import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms


def count_parameters(model):
    """
    Function that counts all learnable parameters in a pytorch model 
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_true_parameters_count(model, alpha, img_size, num_classes):
    """ 
    Function that returns the expected number of parameters for each of the models,
    as given in the paper (Table 3, Table 4, ...)
    """
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
    
