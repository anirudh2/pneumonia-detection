"""Train the model"""

import argparse
import logging
import os
import pdb
import math
import scipy.io as io

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
from DensenetModels import DenseNet121
from DensenetModels import DenseNet201
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import f1_score

# 224x224_CXR
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/balanced', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def computeAUROC(targets, preds):
    auroc_vec = []
    
    try:
        auroc_vec.append(roc_auc_score(targets, preds))
    except:
        pass
    return auroc_vec   
   

# def evaluate_test(model, dataloader, optimizer, scheduler, loss_fn):
    

def evaluate_val(model, dataloader, optimizer, scheduler, loss_fn):

    model.eval()
    
    loss_val = 0
    loss_val_norm = 0
    loss_tensor_mean = 0
    auroc_mean = 0
    counter = 0
    f1_sum = 0
    all_true = np.array([])
    all_pred = np.array([])
    with tqdm(total=len(dataloader)) as t:
        for i, (val_batch, target) in enumerate(dataloader):
            with torch.no_grad():
    #         if params.cuda:
    #             val_batch, target = val_batch.cuda(async=True), target.cuda(async=True)    
                target_batch = torch.zeros(val_batch.size()[0],2)
                target_batch[:,1] = target
                target_batch[:,0] = torch.abs(1-target)

                if params.cuda:
                    val_batch, target_batch = val_batch.cuda(async=True), target_batch.cuda(async=True)

#                 val_batch_in = torch.autograd.Variable(val_batch, volatile=True)
#                 target_in = torch.autograd.Variable(target_batch, volatile=True)
                val_pred = model(val_batch)

                loss_tensor = loss_fn(val_pred, target_batch)
                loss_tensor_mean += loss_tensor

                loss_val += loss_tensor.data[0]
                loss_val_norm += 1
                
                true_curr = np.argmax(target_batch, axis=1)
                pred_curr = np.argmax(val_pred.cpu().numpy(), axis=1)
                
                all_true = np.append(all_true, true_curr)
                all_pred = np.append(all_pred, pred_curr)
                
#                 Find the F1 Score
#                 f1 = computeF1Score(target_batch, val_pred)
#                 if (not math.isnan(f1)):
#                     f1_sum += f1
#                     counter += 1
                
                # Find AUROC
#                 auroc_curr = computeAUROC(target_batch, val_pred, 2)
#                 temp = np.array(auroc_curr).mean()
#                 if (not math.isnan(temp)):
#                     auroc_mean += temp
#                     counter += 1
                t.update()

#     pdb.set_trace()
    auroc_curr = np.array(computeAUROC(all_true, all_pred)).mean()
    f1 = f1_score(all_true, all_pred)
    print('The F1 Score for Val is:', f1)
    loss_norm = loss_val / loss_val_norm
    loss_tensor_norm = loss_tensor_mean / loss_val_norm
    
    return loss_norm, loss_tensor_norm, f1, auroc_curr


def train(model, optimizer, loss_fn, dataloader, metrics, params, scheduler):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    all_true = np.array([])
    all_pred = np.array([])
    
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
#             # move to GPU if available
#             if params.cuda:
#                 train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            #train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss

            labels_batch_rs = torch.zeros(train_batch.size()[0],2)
            labels_batch_rs[:,1] = labels_batch
            labels_batch_rs[:,0] = torch.abs(1-labels_batch)
            
#             pdb.set_trace()
            weights = torch.ones(train_batch.size()[0],2)
            weights[labels_batch==1,:] = 90
            weights[labels_batch==0,:] = 1
            
            # move to GPU if available
            if params.cuda:
                train_batch = train_batch.cuda(async=True)
                labels_batch_rs = labels_batch_rs.cuda(async=True)
                weights_batch = weights.cuda(async=True)
                
            loss_fn = loss_fn = torch.nn.BCELoss(weight = weights_batch, size_average=True)

            train_batch_in = torch.autograd.Variable(train_batch)
            labels_batch_in = torch.autograd.Variable(labels_batch_rs)
            output_batch = model(train_batch_in)
            
            output_pred = output_batch
            
            true_curr = np.argmax(labels_batch_rs, axis=1)
            pred_curr = np.argmax(output_pred.cpu().detach().numpy(), axis=1)
                
            all_true = np.append(all_true, true_curr)
            all_pred = np.append(all_pred, pred_curr)
            
#             pdb.set_trace()
            loss = loss_fn(output_batch, labels_batch_in)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])
#             pdb.set_trace()
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

#     pdb.set_trace()
    f1 = f1_score(all_true, all_pred)
    print('The F1 Score for Train is:', f1)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return loss_avg().data.cpu().numpy()


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, scheduler,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    min_loss = 1000000
    loss_arr = np.array([])
    f1_scores = np.array([])
    auroc_vec = np.array([])
    
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        curr_loss_train = train(model, optimizer, loss_fn, train_dataloader, metrics, params, scheduler)
        loss_arr = np.append(loss_arr, curr_loss_train)
        
        loss_val, loss_tensor, f1, auroc_curr = evaluate_val(model, val_dataloader, optimizer, scheduler, loss_fn)
        f1_scores = np.append(f1_scores, f1)
        auroc_vec = np.append(auroc_vec, auroc_curr)
        
        scheduler.step(loss_tensor.data[0])
        
        loss_val = loss_val.data.cpu().numpy()
        if loss_val < min_loss:
            min_loss = loss_val
        print(loss_val)

#         # Evaluate for one epoch on validation set

#         val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

#         val_acc = val_metrics['accuracy']
#         is_best = val_acc>=best_val_acc

#         # Save weights
#         utils.save_checkpoint({'epoch': epoch + 1,
#                                'state_dict': model.state_dict(),
#                                'optim_dict' : optimizer.state_dict()},
#                                is_best=is_best,
#                                checkpoint=model_dir)

#         # If best_eval, best_save_path
#         if is_best:
#             logging.info("- Found new best accuracy")
#             best_val_acc = val_acc

#             # Save best val metrics in a json file in the model directory
#             best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
#             utils.save_dict_to_json(val_metrics, best_json_path)

#         # Save latest val metrics in a json file in the model directory
#         last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
#         utils.save_dict_to_json(val_metrics, last_json_path)
    io.savemat('data/train_loss.mat', mdict={'loss_arr': loss_arr})
    io.savemat('data/val_f1.mat', mdict={'f1_scores': f1_scores})
    io.savemat('data/val_auroc.mat', mdict={'auroc_vec': auroc_vec})


if __name__ == '__main__':

    nnIsTrained = True

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
#     pdb.set_trace()
    logging.info("- done.")

    # Define the model and optimizer
    if params.cuda:
        model = DenseNet121(params.num_classes, nnIsTrained).cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = DenseNet121(params.num_classes, nnIsTrained)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate, betas=(0.9,0.999),
                           eps=params.eps,weight_decay=params.weight_decay)
    scheduler =  ReduceLROnPlateau(optimizer, factor = params.factor, patience = params.patience, mode = 'min')

    # fetch loss function and metrics
    loss_fn = torch.nn.BCELoss(size_average=True)
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir, scheduler,
                       args.restore_file)
