"""TRAINING FUNCTIONS AND OTHER AUXILIARY FUNCTIONS"""

"""
Author: Leonardo Antiqui <leonardoantiqui@gmail.com>

"""

"""Libraries"""


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import random
import os
import copy
import subprocess
import requests
import matplotlib.pyplot as plt
from collections import OrderedDict

#if os.system("pip show torcheval") != 0:
#    os.system("pip install --quiet torcheval")
    
try:
    from torcheval.metrics import MulticlassAccuracy
    from torcheval.metrics import MulticlassAUROC
except:
    subprocess.check_call(["pip", "install", "--quiet", "torcheval"])
    from torcheval.metrics import MulticlassAccuracy
    from torcheval.metrics import MulticlassAUROC
    

#if os.system("pip show pydrive2") != 0:
#    os.system("pip install --quiet pydrive2")
    
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
except:
    subprocess.check_call(["pip", "install", "--quiet", "pydrive2"])
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials


"""Classification functions"""

# Utility function to train the model
def train_classf(train_dl, model, loss_fn, opt, writer, epoch, l1_lambda=None, clip_value=None, device='cpu'):
    """
    
    The training step in the training loop for a multiclass classification problem 
    
    Args:
        train_dl (torch.utils.data.DataLoader): train data loader
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        opt (torch.optim.Optimizer): optimizer
        writer (torch.utils.tensorboard.writer.SummaryWriter): log writer
        epoch (int): current epoch in the training loop
        l1_lambda (float): weight decay for L1 regularizaton. Default None
        clip_value (float): gradient clipping value. Default None
        device (str): binary value indicating the use of gpu or cpu. Default 'cpu'
    
    Returns:
        model(torch.nn.Module): model
        running_loss (float): current training loss
    
    """
    
    model.train()  #setup model for training. Some types of layers, like batch normalization or dropout, behave differently
    running_loss = 0
    i = 0

    backbone_layers = [name for name, param in model.named_parameters() if 'classifier' not in name and 'fc' not in name]

    # Train with batches of data
    for xb, yb in train_dl:

        xb = xb.to(device)
        yb = yb.to(device)

        # 1. Generate predictions
        Y_hat = model(xb)

        # 2. Reset the gradients to zero
        opt.zero_grad()

        # 3. Calculate loss
        loss = loss_fn(Y_hat, yb)

        # 4. Regularization
        if l1_lambda:
            l1_reg_loss = 0
            for param in model.parameters():
                l1_reg_loss += torch.norm(param, 1)

            loss += l1_lambda * l1_reg_loss

        # 5. Compute gradients
        loss.backward()

        #6. Gradient clipping
        if clip_value:
            backbone_params = [param for name, param in model.named_parameters() if name in backbone_layers]
            #for param in backbone_params:
            #    torch.nn.utils.clip_grad_norm_(param, max_norm=clip_value)
            torch.nn.utils.clip_grad_value_(backbone_params, clip_value=clip_value)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

        #Gradients recording
        if i%(len(train_dl)//5) == 0:
            gradients(model, writer, epoch)

        # 6. Update parameters using gradients
        opt.step()

        #Save loss function values
        i += 1
        running_loss += (1/i)*(loss.detach().item()-running_loss)

        if np.isnan(running_loss):
            print('NaN loss')
            break

    return model, running_loss

def test_classf(test_dl, model, loss_fn, device='cpu'):

    """
    
    Compute the loss function accoring to the prediction made by the model on test dataset
    
    Args:
        test_dl (torch.utils.data.DataLoader): test data loader
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        device (str): binary value indicating the use of gpu or cpu. Default cpu
    
    Returns:
        running_loss (float): current test loss
    
    """
    
    model.eval()
    running_loss = 0
    i = 0

    with torch.no_grad():

        for xb, yb in test_dl:

            xb = xb.to(device)
            yb = yb.to(device)

            # 1. Generate predictions
            Y_hat = model(xb)

            # 2. Calculate loss
            loss = loss_fn(Y_hat, yb)

            #Save loss function values
            i += 1
            running_loss += (1/i)*(loss.detach().item()-running_loss)

            if np.isnan(running_loss):
                print('NaN loss')
                break

    return running_loss


def fit_classf(model, loss_fn, opt, train_dl, test_dl, num_epochs, scheduler=None, writer=None, l1_lambda=None, clip_value=None, device='cpu', print_f=1):
    
    """
    
    The training loop for a multiclass classification problem 
    
    Args:
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        opt (torch.optim.Optimizer): optimizer
        train_dl (torch.utils.data.DataLoader): train data loader
        test_dl (torch.utils.data.DataLoader): test data loader
        num_epochs (int): planned number of epochs for the training loop
        scheduler (torch.optim.lr_scheduler): step size controller. Default None
        writer (torch.utils.tensorboard.writer.SummaryWriter): log writer. Default None
        l1_lambda (float): weight decay for L1 regularizaton. Default None
        clip_value (float): gradient clipping value. Default None
        device (str): binary value indicating the use of gpu or cpu. Default cpu
        print_f (int): printing frequency in epoch number. Default 1
    
    Returns:
        best_model_wts(collections.OrderedDict): dictionary with the parameters of the model with the highest validation accuracy
    
    """

    start_time = time.time() #starting time of the training algorithm

    model.to(device)

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        start_time_epoch = time.time() #starting time of the epoch
        
        #Parameters record
        parameters(model, writer, epoch)

        model, train_loss = train_classf(train_dl, model, loss_fn, opt, writer, epoch, l1_lambda, clip_value, device)
        
        if scheduler:
            scheduler.step()
            
        #Test accuracy
        test_loss = test_classf(test_dl, model, loss_fn, device=device)

        if np.isnan(train_loss):
            break

        #Computing accuracy on train and test datasets
        train_acc = accuracy_classf(model, train_dl, device=device)
        test_acc = accuracy_classf(model, test_dl, device=device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        end_time_epoch = time.time() #end time of the epoch
        time_epoch = end_time_epoch - start_time_epoch
        time_train = end_time_epoch - start_time

        if epoch%print_f == 0:

            print(f'Epoch: {epoch} ',
                  f'time_epoch: {time_epoch:.2f}s ',
                  f'training_loss: {train_loss:.2f} ',
                  f'test_loss: {test_loss:.2f} ',
                  f'training_accuracy: {train_acc:.2f} ',
                  f'test_accuracy: {test_acc:.2f}'
                  )

        if writer:
            writer.add_scalar(tag='training_loss', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
            writer.add_scalar(tag='training_accuracy', scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag='test_accuracy', scalar_value=test_acc, global_step=epoch)
            writer.add_scalar(tag='time_epoch', scalar_value=time_epoch, global_step=epoch)
            writer.add_scalar(tag='time_train', scalar_value=time_train/60, global_step=epoch)
            
        if train_acc > 0.99:
            break

    #Parameters record
    parameters(model, writer, epoch)
    
    if writer:
        writer.flush()

    return best_model_wts

def accuracy_classf(model, data_loader, k=1, device='cpu'):
    
    """
    
    Compute the top-k accuracy of the model on the provided dataset
    
    Args:
        model (torch.nn.Module): model
        data_loader (torch.utils.data.DataLoader): data loader
        k (int): number of highest predictions to consider. Default 1
        device (str): binary value indicating the use of gpu or cpu. Default cpu
    
    Returns:
        accuracy (float): top-k accuracy
    
    """
    
    metric = MulticlassAccuracy(k=k)

    correct_pred = 0
    n = 0

    model.eval() #setup model for evaluation
    metric.reset()
    with torch.no_grad():

        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)

            metric.update(y_logits, y)

    return metric.compute().item()


def auc_classf(model, data_loader, num_classes=10, device='cpu'):

    """
    
    Compute the area under the ROC Curve of the model on the provided dataset
    
    Args:
        model (torch.nn.Module): model
        data_loader (torch.utils.data.DataLoader): data loader
        num_classes (int): number of classes in the dataset. Default 10
        device (str): binary value indicating the use of gpu or cpu. Default cpu
    
    Returns:
        auc (float): area under the ROC Curve
    
    """
    
    metric = MulticlassAUROC(num_classes=num_classes)
    
    correct_pred = 0
    n = 0

    model.eval() #setup model for evaluation
    metric.reset()
    with torch.no_grad():

        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)

            #Multiclass problem
            metric.update(y_logits, y)

    return metric.compute().item()
    


"""Contrastive learning"""
 
def train_contrast(train_dl, model, loss_fn, opt, writer, epoch, l1_lambda=None, clip_value=None, device='cpu'):

    """
    
    The training step in the training loop for a contrastive learning problem 
    
    Args:
        train_dl (torch.utils.data.DataLoader): train data loader
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        opt (torch.optim.Optimizer): optimizer
        writer (torch.utils.tensorboard.writer.SummaryWriter): log writer
        epoch (int): current epoch in the training loop
        l1_lambda (float): weight decay for L1 regularizaton. Default None
        clip_value (float): gradient clipping value. Default None
        device (str): binary value indicating the use of gpu or cpu. Default 'cpu'
    
    Returns:
        model(torch.nn.Module): model
        running_loss (float): current training loss
    
    """
    
    model.train()  #setup model for training. Some types of layers, like batch normalization or dropout, behave differently
    running_loss = 0
    i = 0

    backbone_layers = [name for name, param in model.named_parameters() if 'classifier' not in name and 'fc' not in name]

    # Train with batches of data
    for anchors,positives,negatives,labels in train_dl:

        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        # 1. Generate predictions
        anchor_outs = model(anchors)
        positive_outs = model(positives)
        negative_outs = model(negatives)

        # 2. Reset the gradients to zero
        opt.zero_grad()

        # 3. Calculate loss
        loss = loss_fn(anchor_outs, positive_outs, negative_outs)

        # 4. L1 regularization
        if l1_lambda:
            l1_reg_loss = 0
            for param in model.parameters():
                l1_reg_loss += torch.norm(param, 1)

            loss += l1_lambda * l1_reg_loss

        # 5. Compute gradients
        loss.backward()

        # 6. Gradient clipping
        if clip_value:
            backbone_params = [param for name, param in model.named_parameters() if name in backbone_layers]
            #for param in backbone_params:
            #    torch.nn.utils.clip_grad_norm_(param, max_norm=clip_value)
            torch.nn.utils.clip_grad_value_(backbone_params, clip_value=clip_value)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

        #Gradients recording
        if i%(len(train_dl)//3) == 0:
            gradients(model, writer, epoch)

        # 6. Update parameters using gradients
        opt.step()

        #Save loss function values
        i += 1
        running_loss += (1/i)*(loss.detach().item()-running_loss)

        if np.isnan(running_loss):
            print('NaN loss')
            break

    return model, running_loss


def fit_contrast(model, loss_fn, opt, train_dl, test_dl, num_epochs, scheduler=None, writer=None, l1_lambda=None, clip_value=None, device='cpu', print_f=1):

    """
    
    The training loop for a multiclass classification problem 
    
    Args:
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        opt (torch.optim.Optimizer): optimizer
        train_dl (torch.utils.data.DataLoader): train data loader
        test_dl (torch.utils.data.DataLoader): test data loader
        num_epochs (int): planned number of epochs for the training loop
        scheduler (torch.optim.lr_scheduler): step size controller. Default None
        writer (torch.utils.tensorboard.writer.SummaryWriter): log writer. Default None
        l1_lambda (float): weight decay for L1 regularizaton. Default None
        clip_value (float): gradient clipping value. Default None
        device (str): binary value indicating the use of gpu or cpu. Default cpu
        print_f (int): printing frequency in epoch number. Default 1
    
    Returns:
        best_model_wts(collections.OrderedDict): dictionary with the parameters of the model with the highest validation accuracy
    
    """

    start_time = time.time() #starting time of the training algorithm

    model.to(device)

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        start_time_epoch = time.time() #starting time of the epoch
        
        #Parameters record
        parameters(model, writer, epoch)

        model, train_loss = train_contrast(train_dl, model, loss_fn, opt, writer, epoch, l1_lambda, clip_value, device)
        
        if scheduler:
            scheduler.step()

        if np.isnan(train_loss):
            print('NaN train loss')
            break

        #Computing accuracy on train and test datasets
        train_acc = accuracy_contrast(model, train_dl, device)
        test_acc = accuracy_contrast(model, test_dl, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        end_time_epoch = time.time() #end time of the epoch
        time_epoch = end_time_epoch - start_time_epoch
        time_train = end_time_epoch - start_time

        if print_f:
            if epoch%print_f == 0:

                print(f'Epoch: {epoch} ',
                      f'time_epoch: {time_epoch:.2f}s ',
                      f'training_loss: {train_loss:.2f} ',
                      f'training_accuracy: {train_acc:.2f} ',
                      f'test_accuracy: {test_acc:.2f}'
                      )

        if writer:
            writer.add_scalar(tag='training_loss', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag='training_accuracy', scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag='val_accuracy', scalar_value=test_acc, global_step=epoch)
            writer.add_scalar(tag='time_epoch', scalar_value=time_epoch, global_step=epoch)
            writer.add_scalar(tag='time_train', scalar_value=time_train/60, global_step=epoch)
        
        if train_acc > 0.95:
            break
        
    #Parameters record
    parameters(model, writer, epoch)

    if writer:
        writer.flush()

    return best_model_wts

def accuracy_contrast(model, data_loader, device='cpu'):

    """
    
    Compute the binary accuracy of the model on the provided dataset
    
    Args:
        model (torch.nn.Module): model
        data_loader (torch.utils.data.DataLoader): data loader
        device (str): binary value indicating the use of gpu or cpu. Default cpu
    
    Returns:
        accuracy (float): binary accuracy
    
    """

    correct_pred = 0
    n = 0

    model.eval() #setup model for evaluation
    pdist = torch.nn.PairwiseDistance(p=2)

    with torch.no_grad():

        for anchors,positives,negatives,labels in data_loader:

            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            anchor_outs = model(anchors)
            positive_outs = model(positives)
            negative_outs = model(negatives)

            dis_ap = pdist(anchor_outs, positive_outs)
            dis_an = pdist(anchor_outs, negative_outs)

            n += anchors.size(0)
            # n += labels.numel()
            correct_pred += (dis_ap < dis_an).sum().item()

    return round(correct_pred / n, 4)
    

"""Test accuracy on pool"""
    
def test_acc_pool_func(encoder, MODEL_NAME, EMB_DIM, train_dl_dummy_pool, test_dl_dummy_pool, LOSS, EPOCHS=3, device='cpu'):

    """
    
    Compute the binary accuracy of the model on the provided test dataset after a brief training with train dataset
    
    Args:
        encoder (torch.nn.Module): model
        MODEL_NAME (str): model name
        EMB_DIM (int): embedding dimensionality
        train_dl_dummy_pool (torch.utils.data.DataLoader): train data loader
        test_dl_dummy_pool (torch.utils.data.DataLoader): test data loader
        LOSS (str): type of contrastive loss
        EPOCHS (int): number of epochs for the training loop. Default 3
        device (str): binary value indicating the use of gpu or cpu. Default cpu
    
    Returns:
        test_acc (float): binary accuracy
    
    """

    model = create_model(MODEL_NAME, EMB_DIM)

    state_dict = encoder.state_dict() #to save the original state of the model
    model.load_state_dict(copy.deepcopy(state_dict))

    #Freezing model parameters
    for param in model.parameters():
        param.requires_grad = False

    model = head_change(model, EMB_DIM)

    loss = loss_func(LOSS)
    opt = create_optimizer(model.parameters())

    #print('Computing test accuracy on pool dataset')

    best_model_wts = fit_contrast(model, loss, opt, train_dl_dummy_pool, test_dl_dummy_pool, EPOCHS, device=device, print_f=None)

    model.load_state_dict(best_model_wts)

    test_acc = accuracy_contrast(model, test_dl_dummy_pool, device)

    return test_acc
    
    
"""Parameter and Gradient Recording"""

def gradients(model, writer, epoch):

    """
    
    Capture and register the gradient norm, mean, standard deviation and maximum value of the parameters of the model grouped by layer
    
    Args:
        model (torch.nn.Module): model
        writer (torch.utils.tensorboard.writer.SummaryWriter): log writer
        epoch (int): current epoch in the training loop
    
    """

    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    for name, g in grads:
        norm = torch.linalg.vector_norm(g, ord=2).item()
        mean = torch.mean(g).item()
        std = torch.std(g).item()
        max = torch.max(torch.abs(g)).item()

        if writer:
            writer.add_scalar(tag='Grad_norm_'+name, scalar_value=norm, global_step=epoch)
            writer.add_scalar(tag='Grad_mean_'+name, scalar_value=mean, global_step=epoch)
            writer.add_scalar(tag='Grad_std_'+name, scalar_value=std, global_step=epoch)
            writer.add_scalar(tag='Grad_max_'+name, scalar_value=max, global_step=epoch)


def parameters(model, writer, epoch):

    """
    
    Capture and register the parameter norm, mean, standard deviation and maximum value of the layers of the model
    
    Args:
        model (torch.nn.Module): model
        writer (torch.utils.tensorboard.writer.SummaryWriter): log writer
        epoch (int): current epoch in the training loop
    
    """

    for name, p in model.named_parameters():
        norm = torch.linalg.vector_norm(p, ord=2).item()
        mean = torch.mean(p).item()
        std = torch.std(p).item()
        max = torch.max(torch.abs(p)).item()

        if writer:
            writer.add_scalar(tag='Param_norm_'+name, scalar_value=norm, global_step=epoch)
            writer.add_scalar(tag='Param_mean_'+name, scalar_value=mean, global_step=epoch)
            writer.add_scalar(tag='Param_std_'+name, scalar_value=std, global_step=epoch)
            writer.add_scalar(tag='Param_max_'+name, scalar_value=max, global_step=epoch)
        

"""Auxiliary functions"""

"""Model"""

def head_change(model, EMB_DIM=100):

    """
    
    Change the number of neurons in the last layer of the head of the CNN
    
    Args:
        model (torch.nn.Module): model
        EMB_DIM (int): embedding dimensionality of the head. Default 100
        
    Returns:
        model (torch.nn.Module): model
        
    """
    
    model_name = model.name
    
    if model_name == 'ResNet18':
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=EMB_DIM, bias=False)
        
    elif model_name == 'EfficientNetB1':
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=EMB_DIM, bias=False)
        
    elif model_name == 'MobileNetV2':
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=EMB_DIM, bias=False)
        
    elif model_name == 'SqueezeNet1':
        model.classifier[1] = torch.nn.Conv2d(in_channels=model.classifier[1].in_channels, out_channels=EMB_DIM, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
    elif model_name == 'AlexNet':
        model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=EMB_DIM, bias=False)
        
    elif model_name == 'VGG11':
        model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=EMB_DIM, bias=False)
        
    elif model_name == 'ViTB16':
        model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=EMB_DIM, bias=False)
        
    elif model_name == 'MobileNetV3':
        model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=EMB_DIM, bias=False)
        
    return model


def create_model(model_name, EMB_DIM=100, weights=None):

    """
    
    Create a CNN with the weights specified parameter weights
    
    Args:
        model_name (str): model name
        EMB_DIM (int): embedding dimensionality of the head. Default 100
        weights (str or collections.OrderedDict): network weights
        
    Returns:
        model (torch.nn.Module): model
        
    """

    if model_name == 'ResNet18':
        
        if isinstance(weights, OrderedDict):
            model = torchvision.models.resnet18(weights=None)
            EMB_DIM = weights['fc.weight'].size()[0]
        else:
            model = torchvision.models.resnet18(weights=weights)
        
    elif model_name == 'EfficientNetB1':
        if isinstance(weights, OrderedDict):
            model = torchvision.models.efficientnet_b1(weights=None)
            EMB_DIM = weights['classifier.1.weight'].size()[0]
        else:
            model = torchvision.models.efficientnet_b1(weights=weights)
        
    elif model_name == 'MobileNetV2':
        model = torchvision.models.mobilenet_v2(weights=weights)
        

    elif model_name == 'SqueezeNet1':
        if isinstance(weights, OrderedDict):
            model = torchvision.models.squeezenet1_0(weights=None)
            EMB_DIM = weights['classifier.1.weight'].size()[0]
        else:
            model = torchvision.models.squeezenet1_0(weights=weights)
        
    elif model_name == 'AlexNet':
        if isinstance(weights, OrderedDict):
            model = torchvision.models.alexnet(weights=None)
            EMB_DIM = weights['classifier.6.weight'].size()[0]
        else:
            model = torchvision.models.alexnet(weights=weights)
        
    elif model_name == 'VGG11':
        if isinstance(weights, OrderedDict):
            model = torchvision.models.vgg11(weights=None)
            EMB_DIM = weights['classifier.6.weight'].size()[0]
        else:
            model = torchvision.models.vgg11(weights=weights)
            
    elif model_name == 'ViTB16':
        if isinstance(weights, OrderedDict):
            model = torchvision.models.vit_b_16(weights=None)
            EMB_DIM = weights['heads.head.weight'].size()[0]
        else:
            model = torchvision.models.vit_b_16(weights=weights)
            
    elif model_name == 'MobileNetV3':
        if isinstance(weights, OrderedDict):
            model = torchvision.models.mobilenet_v3_small(weights=None)
            EMB_DIM = weights['classifier.3.weight'].size()[0]
        else:
            model = torchvision.models.mobilenet_v3_small(weights=weights)
            
    elif model_name == 'custom':
        model = torch.nn.Sequential(
              torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=2, stride=2),
              torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=2, stride=2),
              torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
              torch.nn.ReLU(inplace=True),
              torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
              torch.nn.Flatten(),
              torch.nn.ReLU(),
              torch.nn.Linear(in_features=64, out_features=EMB_DIM)
            )
        
    model.name = model_name
    model.emb_dim = EMB_DIM
    
    model = head_change(model, EMB_DIM)

    return model


"""Optimizer"""

def create_optimizer(parameters, name='SGD', LR=0.01, extra=None):

    """
    
    Create an optimizer of the specified type
    
    Args:
        parameters (list): list with model's parameters
        name (str): optimizer type
        LR (float): learning rate
        extra (list): extra arguments for momentum, L2 weight decay and Nesterov mode
        
    Returns:
        opt (torch.optim.Optimizer): optimizer
        
    """

    if name == 'SGD':

        if extra is None:
            MOMENTUM = 0
            DECAY = 0
            NESTEROV = False
        else:
            MOMENTUM = extra[0]
            DECAY = extra[1]
            NESTEROV = extra[2]

        #Stochastic Gradient Descent
        opt = torch.optim.SGD(
            parameters,
            lr=LR, #learning rate
            momentum=MOMENTUM, #momentum factor
            weight_decay=DECAY, #weight decay (L2 penalty)
            dampening=0, #dampening for momentum
            nesterov=NESTEROV #enables Nesterov momentum
            )
    
    elif name == 'Adam':

        #Adam
        opt = torch.optim.Adam(
            parameters,
            lr=LR, #learning rate
            betas=(0.9, 0.999), #coefficients used for computing running averages of gradient and its square
            weight_decay=0 #weight decay (L2 penalty)
            )
    
    elif name == 'AdamW':

        #AdamW
        opt = torch.optim.AdamW(
            parameters,
            lr=LR, #learning rate
            betas=(0.9, 0.999), #coefficients used for computing running averages of gradient and its square
            weight_decay=0 #weight decay (L2 penalty)
            )

    elif name == 'RMSProp':

        #RMSProp
        opt = torch.optim.RMSprop(
            parameters,
            lr=LR, #learning rate
            alpha=0.99, #coefficients used for computing running averages of gradient and its square
            weight_decay=0 #weight decay (L2 penalty)
            )

    return opt
    
    
"""Downloading target datasets"""


def dataloader(DATASET, IMG_SIZE, BATCH_SIZE, SUBSET_SIZE=None):

    """
    
    Create a dataloader with the specified dataset
    
    Args:
        DATASET (str): target dataset name
        IMG_SIZE (int): desired image size
        BATCH_SIZE (int): batch size
        SUBSET_SIZE (int): number of data points in the train data loader
        
    Returns:
        train_loader (torch.utils.data.DataLoader): train data loader
        test_loader (torch.utils.data.DataLoader): test data loader
        
    """

    transforms = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                    torchvision.transforms.ToTensor(), #to transform the matrices in tensors
                                                    # torchvision.transforms.Normalize(mean, std)
                                                ])

    if DATASET == 'Flowers102':

        # download and create datasets
        train_dataset = torchvision.datasets.Flowers102(root='./datasets',
                                              transform=transforms,
                                              split='train',
                                              download=True)

        test_dataset = torchvision.datasets.Flowers102(root='./datasets',
                                              transform=transforms,
                                              split='test',
                                              download=True)
                                              
    elif DATASET[:7] == 'EuroSAT':
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        # download and create datasets
        dataset = torchvision.datasets.EuroSAT(root='./datasets',
                                              transform=transforms,
                                              download=True)
        
        #We split the train dataset into a train and test dataset
        split_ratio = 0.8
        N = len(dataset)
        N_train = int(N*split_ratio)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [N_train, N-N_train])

    elif DATASET[:4] == 'SVHN':
        
        # download and create datasets
        train_dataset = torchvision.datasets.SVHN(root='./datasets/SVHN',
                                              transform=transforms,
                                              split='train',
                                              download=True)

        test_dataset = torchvision.datasets.SVHN(root='./datasets/SVHN',
                                              transform=transforms,
                                              split='test',
                                              download=True)
                                              
    elif DATASET[:8] == 'CIFAR100':

        # download and create datasets
        train_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                              transform=transforms,
                                              train=True,
                                              download=True)

        test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                              transform=transforms,
                                              train=False,
                                              download=True)
    
    elif DATASET[:7] == 'CIFAR10':

        # download and create datasets
        train_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                              transform=transforms,
                                              train=True,
                                              download=True)

        test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                              transform=transforms,
                                              train=False,
                                              download=True)
        
    

    #Subset
    if SUBSET_SIZE:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        subset_indices = random.sample(indices, SUBSET_SIZE)
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=2)
    
    return train_loader, test_loader

    
    
"""Feature visualization"""

def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook

def filter_vizs(model, outlayer_name, num_epochs, n_filters, ncols, img_size, title):

    """
    
    Plot a chart showing the input that maximize the output of filter in the CNN model
    Based on the activation maximization optimization
    
    Args:
        model (torch.nn.Module): model
        outlayer_name (str): name of the layer in the CNN
        num_epochs (int): number of epochs for the optimization process
        n_filters (int): number of filters in the CNN layer to plot
        ncols (int): number of columns for the chart
        img_size (int): input image size to consider
        title (str): title for the chart
    
    """

    nrows = int(n_filters/ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,8))

    fig.suptitle(title + '_' + outlayer_name)

    # model.cpu()
    DEVICE = next(model.parameters()).device
    model.eval()

    act_dict = {}
    for name, module in model.named_modules():
        if name == outlayer_name:
            module.register_forward_hook(layer_hook(act_dict, outlayer_name))

    for f_idx in range(n_filters):

        x = torch.rand(img_size, requires_grad=True, device=DEVICE)
        opt = torch.optim.Adam([x], lr=0.1, weight_decay=1e-6)

        for _ in range(num_epochs):
            opt.zero_grad()

            model(x)
            layer_out = act_dict[outlayer_name]
            h = -layer_out[:,f_idx].mean()

            h.backward()
            opt.step()

        #Post-processing
        filter = x[0].detach().permute(1, 2, 0).numpy()
        # filter = (filter - np.min(filter)) / (np.max(filter) - np.min(filter)) #Min-Max Normalization
        filter = np.clip((filter - np.mean(filter)) / np.std(filter), 0, 1)

        ax[int(f_idx/ncols)][f_idx%ncols].set_title(f'Filter : {f_idx}')
        ax[int(f_idx/ncols)][f_idx%ncols].imshow(filter)
        ax[int(f_idx/ncols)][f_idx%ncols].axes.yaxis.set_visible(False)
        ax[int(f_idx/ncols)][f_idx%ncols].axes.xaxis.set_visible(False)

    plt.ioff()
    plt.tight_layout(pad=1)
    plt.show()
  
  
"""Tensorboard file reader"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(file_path):

    """
    
    Extract the recorded values in the Tensorboard event file
    
    Args:
        file_path (str): file path
        
    Returns:
        data (dict): dictionary with the event data
    
    """

    event_acc = EventAccumulator(file_path)
    event_acc.Reload()

    # Extracting scalar data
    tags = event_acc.Tags()['scalars']  # Get all scalar tags
    data = {}

    for tag in tags:
        data[tag] = {'step': [], 'value': []}
        for event in event_acc.Scalars(tag):
            data[tag]['step'].append(event.step)
            data[tag]['value'].append(event.value)

    return data
    
    
"""Dataset statistics"""    

def dataset_statistics(dataset):

    """
    
    Compute the mean and standard deviation for the pixels in images of a dataset
    
    Args:
        dataset (torch.utils.data.Dataset): dataset
        
    Returns:
        mean (float): mean
        std (float): standard deviation
    
    """

    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    count = 0

    # Iterate over the dataset to compute mean and std
    for image, _ in dataset:
        # Accumulate sum for mean calculation
        mean_sum += torch.mean(image, dim=(1, 2))
        
        # Accumulate sum of squares for std calculation
        std_sum += torch.std(image, dim=(1, 2), unbiased=False)
        
        count += 1

    # Calculate mean and std
    mean = mean_sum / count
    std = std_sum / count

    return mean, std
    
    
"""SRP"""

def shrink_perturb(net, shrink, perturb):

    """
    
    Apply shrink and perturb over the parameters of the network
    Based on the technique presented in the paper "On Warm-Starting Neural Network Training" by Ash and Adams
    
    Args:
        net (torch.nn.Module): network
        shrink (float): shrinking coefficient
        perturb (float): pertubation coefficient
    
    """

    # using a randomly-initialized model as a noise source respects how different kinds 
    # of parameters are often initialized differently
    
    model = create_model(net.name, EMB_DIM=net.emb_dim, weights=None)
    
    DEVICE = next(net.parameters()).device
    model.to(DEVICE)

    params1 = model.parameters()
    params2 = net.parameters()
    for p1, p2 in zip(*[params1, params2]):
        p2.data = copy.deepcopy(shrink * p2.data + perturb * p1.data)
        

"""Partial reinitialiation"""

def param_reinitialization(model, n):

    """
    
    Reinitialize the parameter of CNNs per layer type 
    
    Args:
        model (torch.nn.Module): model
        n (int): number of Conv2d layers to keep from the bottom of the network
    
    """

    i = 0
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            if i > n:
                # torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            i += 1
        
        elif isinstance(module, torch.nn.BatchNorm2d):
            # Reinitialize weights and biases of batch normalization layer
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, torch.nn.Linear):
            # Reinitialize weights and biases of linear layer
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
                
"""Triplet losses"""

#def cosine_euclidean(x1, x2, P=2):
#    return torch.nn.functional.pairwise_distance(x1, x2, p=P) - torch.nn.functional.cosine_similarity(x1, x2, dim=1)
    
class cosine_euclidean(torch.nn.Module):

    """
    
    Distance metric between vectors composed of the norm minus the cosine similarity
    
    Atributes:
        p (int): norm
    
    """

    def __init__(self, p):
        super(cosine_euclidean, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        # Compute your custom loss

        loss = torch.nn.functional.pairwise_distance(x1, x2, p=self.p) - torch.nn.functional.cosine_similarity(x1, x2, dim=1)

        return loss

class NCELoss(torch.nn.Module):

    """
    
    Noise Contrastive Estimation loss
    
    Atributes:
        temp (float): temperature
    
    """

    def __init__(self, temp):
        super(NCELoss, self).__init__()
        self.temp = temp

    def forward(self, anchor, positive, negative):
        # Compute your custom loss

        pos_scores = torch.nn.functional.cosine_similarity(anchor, positive, dim=1)
        neg_scores = torch.nn.functional.cosine_similarity(anchor, negative, dim=1)

        loss = -torch.log(torch.exp(pos_scores/self.temp) / (torch.exp(pos_scores/self.temp) + torch.exp(neg_scores/self.temp)))

        return torch.mean(loss)
        
def loss_func(LOSS, MARGIN=1.0):

    """
    
    Extract the recorded values in the Tensorboard event file
    
    Args:
        LOSS (str): contrastive loss name + hyperparmeter
        MARGIN (float): margin for triplet loss
        
    Returns:
        loss (torch.nn.Module): loss function
    
    """
    
    if LOSS[:4] == 'Norm':
    
        P = eval(LOSS[4:])
        
        loss = torch.nn.TripletMarginLoss(
            margin=MARGIN,
            p=P, #The norm degree for pairwise distance.
            eps=10**(-6), #Small constant for numerical stability
            swap=False, #distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses
            reduction='mean' #Specifies the reduction to apply to the output
        )
        
    elif LOSS == 'Cosine':

        loss = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=-torch.nn.CosineSimilarity(dim=1),
            margin=MARGIN
        )
        
    elif LOSS[:11] == 'Cosine+Norm':
    
        P = eval(LOSS[11:])
    
        loss = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=cosine_euclidean(P),
            margin=MARGIN
        )
        
    elif LOSS[:3] == 'NCE':
    
        T = eval(LOSS[3:])
        
        loss = NCELoss(T)
        
    return loss
    
    
"""Speed convergence function"""

def trapezoidal_integral(data):

    """
    
    Integral by trapezoidal rule
    
    Args:
        data (list): list of points of the learning curve
        
    Returns:
        integral (float): area under the list of points
    
    """
    
    n = len(data)
    integral = 0.0
    for i in range(1, n):
        integral += (data[i-1] + data[i]) / 2.0
    return integral

def converge_diff(target_file, reference_file, experiment_path):

    """
    
    Area between target learning curve and the reference learning curve
    
    Args:
        target_file (str): name of the target file
        reference_file (str): name of the reference file
        experiment_path (str): path of the folder with Tensorboard files
        
    Returns:
        convergence (float): area between the learning curves
    
    """

    file_paths = [file for file in os.listdir(experiment_path + 'Logs') if file.split(".")[-1] == target_file]
    data = extract_scalars(experiment_path + 'Logs/' + file_paths[0])
    test_accs1 = data['test_accuracy']['value']
    

    file_paths = [file for file in os.listdir(experiment_path + 'Logs') if file.split(".")[-1] == reference_file]
    data = extract_scalars(experiment_path + 'Logs/' + file_paths[0])
    test_accs2 = data['test_accuracy']['value']
    
    if len(test_accs1) == len(test_accs2):
        integral1 = trapezoidal_integral(test_accs1)
        integral2 = trapezoidal_integral(test_accs2)
        
    else:
        length = min(len(test_accs1), len(test_accs2))
        integral1 = trapezoidal_integral(test_accs1[:length])
        integral2 = trapezoidal_integral(test_accs2[:length])

    return round(integral1-integral2, 3)

                
"""PyDrive functions"""

def check_internet_connection():

    """
    
    Check the internet connection status
    
    Returns:
        
        status (bool): internet connection status
    
    """

    try:
        # Attempt to make a GET request to a known website
        response = requests.get("http://www.google.com", timeout=5)
        # Check if the response status code indicates success (2xx)
        if response.status_code == 200:
            return True
        else:
            return False
    except:
        return False

def get_code_by_name(drive, title):

    """
    
    Download the desired py file with the corresponding code
    
    Args:
        drive (pydrive2.drive.GoogleDrive)
        title: name of the py file
    
    """

    if check_internet_connection():
        file_path = title
        if not os.path.exists(file_path):
            foldered_list=drive.ListFile({'q': "'1UHTbF3RmzTjmoEGQ_4dcGRPFe1qtnAZC' in parents and trashed=false"}).GetList()
            for file in foldered_list:
                if(file['title']==title):
                    model = drive.CreateFile({'id': file['id']})
                    model.GetContentFile(file_path,
                                        # remove_bom=False
                                        )
                    print(f'{title} donwloaded!')

def get_model_by_name(drive, title):

    """
    
    Download the desired pth file with the corresponding model parameters
    
    Args:
        drive (pydrive2.drive.GoogleDrive)
        title: name of the pth file
    
    """

    if check_internet_connection():
        file_path = 'Models/' + title
        if not os.path.exists(file_path):
            foldered_list=drive.ListFile({'q': "'1Y-5keO4kvYomaBeWcJE7RfliL84kM9-I' in parents and trashed=false"}).GetList()
            for file in foldered_list:
                if(file['title']==title):
                    model = drive.CreateFile({'id': file['id']})
                    model.GetContentFile(file_path,
                                        # remove_bom=False
                                        )
                    print(f'{title} donwloaded!')

def get_curricula_by_name(drive, title):

    """
    
    Download the desired txt file with the corresponding curriculum
    
    Args:
        drive (pydrive2.drive.GoogleDrive)
        title: name of the txt file
    
    """

    if check_internet_connection():
        file_path = 'Curricula/' + title
        if not os.path.exists(file_path):
            foldered_list=drive.ListFile({'q': "'1yp6KlEapGcfZYlR6_6IzYI8WV1-w2CIi' in parents and trashed=false"}).GetList()
            for file in foldered_list:
                if(file['title']==title):
                    model = drive.CreateFile({'id': file['id']})
                    model.GetContentFile(file_path,
                                        # remove_bom=False
                                        )
                    print(f'{title} donwloaded!')
                
def get_log_by_name(drive, title):

    """
    
    Download the desired event file with the corresponding training logs
    
    Args:
        drive (pydrive2.drive.GoogleDrive)
        title: name of event file
    
    """

    if check_internet_connection():
        file_exist = any(title in filename for filename in os.listdir('Logs'))
        if not file_exist:
            foldered_list=drive.ListFile({'q': "'1cSHXtwsEIg6jtt68OLrpkYl4IDhRDDtH' in parents and trashed=false"}).GetList()
            for file in foldered_list:
                if(file['title'].split(".")[-1]==title):
                    file_path = 'Logs/' + file['title']
                    model = drive.CreateFile({'id': file['id']})
                    model.GetContentFile(file_path,
                                        # remove_bom=False
                                        )
                    print(f'{title} donwloaded!')

def upload_model_by_name(drive, title):

    """
    
    Upload the desired pth file with the corresponding model parameters
    
    Args:
        drive (pydrive2.drive.GoogleDrive)
        title: name of pth file
    
    """

    if check_internet_connection():
        metadata = {
            'parents': [
                {"id": '1Y-5keO4kvYomaBeWcJE7RfliL84kM9-I'}
            ],
            'title': title,
            'mimeType': 'application/x-zip'
        }

        file_path = 'Models/' + title
        if os.path.exists(file_path):
            model = drive.CreateFile(metadata=metadata)
            # Read file and set it as a content of this instance
            model.SetContentFile(file_path)
            model.Upload() # Upload the file
            print(f'{title} uploaded!')
        
def upload_log_by_name(drive, title):

    """
    
    Upload the desired event file with the corresponding training logs
    
    Args:
        drive (pydrive2.drive.GoogleDrive)
        title: name of event file
    
    """

    if check_internet_connection():
        metadata = {
            'parents': [
                {"id": '1cSHXtwsEIg6jtt68OLrpkYl4IDhRDDtH'}
            ],
            'title': title,
            'mimeType': 'application/octet-stream'
        }

        file_path = 'Logs/' + title
        if os.path.exists(file_path):
            model = drive.CreateFile(metadata=metadata)
            # Read file and set it as a content of this instance
            model.SetContentFile(file_path)
            model.Upload() # Upload the file
            print(f'{title} uploaded!')
        
        
def get_id_of_title(drive, title, parent_directory_id):
  foldered_list=drive.ListFile({'q':  "'"+parent_directory_id+"' in parents and trashed=false"}).GetList()
  for file in foldered_list:
    if(file['title']==title):
      return file['id']
    return None
    
"""Reference event files"""

"""Dictionary with hyperparameters for each tuple Dataset-Network"""
configs = {
    'Flowers102': {
        'n_classes': 102,
        'img_size': 128,
        'batch_size': 64,
        'subset': None,
        'ResNet18':{
            'n_epochs': 15,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.27,
            'ref_file': 'ResNet18_Flowers102_20240203_1431'
        },
        'EfficientNetB1':{
            'n_epochs': 15,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.13,
            'ref_file': 'EfficientNetB1_Flowers102_20240125_1122'
        },
        'SqueezeNet1':{
            'n_epochs': 15,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.040,
            'ref_file': 'SqueezeNet1_Flowers102_20240122_1410'
        },
    },
    'EuroSAT': {
        'n_classes': 10,
        'img_size': 64,
        'batch_size': 64,
        'subset': None,
        'ResNet18':{
            'n_epochs': 11,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.85,
            'ref_file': 'ResNet18_EuroSAT_20240122_1017'
        },
        'EfficientNetB1':{
            'n_epochs': 11,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.91,
            'ref_file': 'EfficientNetB1_EuroSAT_20240122_1042'
        },
        'SqueezeNet1':{
            'n_epochs': 31,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.86,
            'ref_file': 'SqueezeNet1_EuroSAT_20240311_1606'
        },
    },
    'EuroSAT(5400)': {
        'n_classes': 10,
        'img_size': 64,
        'batch_size': 64,
        'subset': 5400,
        'ResNet18':{
            'n_epochs': 19,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.78,
            'ref_file': 'ResNet18_EuroSAT_20240209_1538'
        },
        'EfficientNetB1':{
            'n_epochs': 19,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'max_top1': 0.76,
            'ref_file': 'EfficientNetB1_EuroSAT_20240307_1552'
        },
        'SqueezeNet1':{
            'n_epochs': 19,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.72,
            'ref_file': 'SqueezeNet1_EuroSAT_20240308_1652'
        },
    },
    'SVHN': {
        'n_classes': 10,
        'img_size': 64,
        'batch_size': 64,
        'subset': None,
        'ResNet18':{
            'n_epochs': 7,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.92,
            'ref_file': 'ResNet18_SVHN_20240122_1807'
        },
        'EfficientNetB1':{
            'n_epochs': 9,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.92,
            'ref_file': 'EfficientNetB1_SVHN_20240122_1854'
        },
        'SqueezeNet1':{
            'n_epochs': 9,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.93,
            'ref_file': 'SqueezeNet1_SVHN_20240122_1830'
        },
    },
    'SVHN(5000)': {
        'n_classes': 10,
        'img_size': 64,
        'batch_size': 64,
        'subset': 5000,
        'ResNet18':{
            'n_epochs': 13,
            'learning_rate': 0.001,
            'weight_decay': 0.1,
            'max_top1': 0.77,
            'ref_file': 'ResNet18_SVHN_20240224_0931'
        },
        'EfficientNetB1':{
            'n_epochs': 19,
            'learning_rate': 0.001,
            'weight_decay': 0.1,
            'max_top1': 0.73,
            'ref_file': 'EfficientNetB1_SVHN_20240307_1427'
        },
        'SqueezeNet1':{
            'n_epochs': 31,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.85,
            'ref_file': 'SqueezeNet1_SVHN_20240311_1337'
        },
    },
    'CIFAR10': {
        'n_classes': 10,
        'img_size': 64,
        'batch_size': 64,
        'subset': None,
        'ResNet18':{
            'n_epochs': 15,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.77,
            'ref_file': 'ResNet18_CIFAR10_20240428_1937'
        },
        'EfficientNetB1':{
            'n_epochs': 15,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.8,
            'ref_file': 'EfficientNetB1_CIFAR10_20240428_1948'
        },
        'SqueezeNet1':{
            'n_epochs': 15,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.9,
            'ref_file': 'SqueezeNet1_CIFAR10_20240428_2007'
        },
    },
    'CIFAR10(5000)': {
        'n_classes': 10,
        'img_size': 64,
        'batch_size': 64,
        'subset': 5000,
        'ResNet18':{
            'n_epochs': 15,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.48,
            'ref_file': 'ResNet18_CIFAR10_20240306_1725'
        },
        'EfficientNetB1':{
            'n_epochs': 15,
            'learning_rate': 0.001,
            'weight_decay': 0.1,
            'max_top1': 0.46,
            'ref_file': 'EfficientNetB1_CIFAR10_20240307_1504'
        },
        'SqueezeNet1':{
            'n_epochs': 19,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.4,
            'ref_file': 'SqueezeNet1_CIFAR10_20240308_1732'
        },
    },
    'CIFAR100': {
        'n_classes': 100,
        'img_size': 64,
        'batch_size': 64,
        'subset': None,
        'ResNet18':{
            'n_epochs': 11,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.46,
            'ref_file': 'ResNet18_CIFAR100_20240122_0708'
        },
        'EfficientNetB1':{
            'n_epochs': 11,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.47,
            'ref_file': 'EfficientNetB1_CIFAR100_20240122_0759'
        },
        'SqueezeNet1':{
            'n_epochs': 11,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'max_top1': 0.29,
            'ref_file': 'SqueezeNet1_CIFAR100_20240122_0730'
        },
    },
    'CIFAR100(5000)': {
        'n_classes': 100,
        'img_size': 64,
        'batch_size': 64,
        'subset': 5000,
        'ResNet18':{
            'n_epochs': 21,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'max_top1': 0.18,
            'ref_file': 'ResNet18_CIFAR100(5000)_20240428_1926'
        },
        'EfficientNetB1':{
            'n_epochs': 21,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.1,
            'ref_file': 'EfficientNetB1_CIFAR100(5000)_20240428_1929'
        },
        'SqueezeNet1':{
            'n_epochs': 21,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_top1': 0.1,
            'ref_file': 'SqueezeNet1_CIFAR100(5000)_20240428_1934'
        },
    },
}