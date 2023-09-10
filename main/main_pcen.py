# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:56:58 2023

@author: Spencer perkins

Main script for K-fold cross validation with trainable Per-channel Energy Normalization frontend
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import copy

from datetime import datetime
import sys

sys.path.insert(1, '../scripts_utils/')
from get_data_kfold import getKfoldDATA
from birdhouse_dataset import birdhouseDataset

sys.path.insert(1, '../scripts_nets/')
from birdhouse_l3 import birdHouseL3

sys.path.insert(1, '../scripts_frontends/')
from pcen import PCEN

#%% Gradient anomaly detection
torch.autograd.set_detect_anomaly(True)
# For result data title
rd_fold = ['1st',
         '2nd',
         '3rd',
         '4th',
         '5th',
         '6th',
         '7th',
         '8th',
         '9th',
         '10th']

for i in range(10):
    folds = ['1st.csv',
             '2nd.csv',
             '3rd.csv',
             '4th.csv',
             '5th.csv',
             '6th.csv',
             '7th.csv',
             '8th.csv',
             '9th.csv',
             '10th.csv']
    val_fold = folds[i]
    #%% Get Data
    audio_path, tr_meta, vl_meta = getKfoldDATA.get_folded_data(val_fold=val_fold, folds=folds)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    features = 'melspec'
    print(f'Feature Extraction: {features}')
    train_data = birdhouseDataset(label_file=tr_meta, audio_dir=audio_path,
                                  features=features
                                  )
    val_data = birdhouseDataset(label_file=vl_meta, audio_dir=audio_path,
                                features=features
                                )
    #%% Split data/ define dataloaders

    # Parameters
    train_batch = 20
    val_batch = 20
    lr =  0.0001
    epochs = 75

    # Data loaders
    trainLoader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=train_batch,
                                              shuffle=True
                                              )
    valLoader = torch.utils.data.DataLoader(dataset=val_data,
                                            batch_size=val_batch,
                                            shuffle=True
                                            )

    # Set Model and training parameters
    # Calculate steps per epoch for training and validation set
    trainSteps = len(trainLoader.dataset) // train_batch
    valSteps = len(valLoader.dataset) // val_batch

    t_val = 2**5
    model_frontend = PCEN(n_bands=128, t_val=t_val, alpha=0.8,
                              delta=10., r=0.25,
                              eps=10e-6)
    print('Number of frontend params: {}'.format(
        sum([p.data.nelement() for p in model_frontend.parameters()])))
    model_cnn = birdHouseL3()
    print('\nNumber of CNN params: {}\n-------------------------\n'.format(
        sum([p.data.nelement() for p in model_cnn.parameters()])))

    model_frontend.to(device)
    model_cnn.to(device)

    # Model weights data
    best_front_wts = copy.deepcopy(model_frontend.state_dict())
    best_cnn_wts = copy.deepcopy(model_cnn.state_dict())
    best_acc = 0.0

    # Optimizer
    params = list(model_frontend.parameters())+list(model_cnn.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=1e-3)

    # Define Loss function
    loss_fun = nn.BCELoss()

    #%% Training
    # Dictionary to store training history
    hist = {'train_loss': np.zeros(epochs, dtype=float),
            'train_acc': np.zeros(epochs, dtype=float),
            'val_loss': np.zeros(epochs, dtype=float),
            'val_acc': np.zeros(epochs, dtype=float),
            'no_bird_acc': np.zeros(epochs, dtype=float),
            'has_bird_acc': np.zeros(epochs, dtype=float),
            'F1': np.zeros(epochs, dtype=float),
            't_value': np.zeros(epochs, dtype=float),
            'Time':[]
            }

    print('[INFO] Training the network...')
    startTime = time.time()

    countT = trainSteps
    countV = valSteps
    print(f'Training steps: {countT}')
    print(f'Validation steps: {countV}\n-----------------------\n')

    stop_pred = 0 # Variable to check if predictions are defaulting to 1 or 0 and stop
    no_stop = 0  # Variable for comparing number of identical predictions vs non-identical
    # Loop over our epochs
    for e in range(0, epochs):
        # Set the model in training mode
        model_frontend.train()
        model_cnn.train()
        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # Initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        epoch_start = time.time()
        iter_count = 0
        print(f'\n---Training epoch {e+1}---\n')
        # Loop over the training set
        for (x, y) in trainLoader:
            # Send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # Perform a forward pass and calculate the training loss
            front = model_frontend(x)
            pred = model_cnn(front)
            pred = pred.to(torch.float32)
            apred = pred.cpu().detach().numpy()
            predicts = [p for p in apred]
            if len(set(predicts)) == 1:
                stop_pred += 1
                print('TRUE ALL VALUES EQUAL FOR PREDICTION')
            else:
                no_stop += 1
            if stop_pred == 50:
                total_iters = stop_pred + no_stop
                print('50 iterations with equal predictions reached')
                print(f'Total iterations Reached = {total_iters}')
                stop() # STOP TRAINING IF 50 iteration of identical predictions
            y = y.to(torch.float32)
            loss = loss_fun(pred, y)
            # Zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.round() == y).type(
                torch.float).sum().item()
            if countT % 10 == 0:
                print(f'\n-----------\n {trainSteps-countT}th/{trainSteps} iteration complete')
            countT -= 1
            iter_count += 1

        # Switch off autograd for evaluation
        with torch.no_grad():
            val_cm = {'pred': torch.zeros(0,dtype=torch.float32, device='cpu'),
                      'truth': torch.zeros(0,dtype=torch.float32, device='cpu')}
            # Set the model in evaluation mode
            model_frontend.eval()
            model_cnn.eval()
            print(f'\n---Validation epoch {e+1}---\n')
            # Loop over the validation set
            for (x, y) in valLoader:
                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # Make the predictions and calculate the validation loss
                front = model_frontend(x)
                pred = model_cnn(front)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                totalValLoss += loss_fun(pred, y)
                # Calculate the number of correct predictions
                valCorrect += (pred.round() == y).type(
                    torch.float).sum().item()
                val_cm['pred']=torch.cat([val_cm['pred'],pred.round().view(-1).cpu()])
                val_cm['truth']=torch.cat([val_cm['truth'],y.view(-1).cpu()])
                if countV % 5 == 0:
                    print(f'\n-----------\n {valSteps-countV}th/{valSteps} iteration complete')
                epoch_acc = valCorrect / len(trainLoader.dataset)
                if epoch_acc > best_acc:
                    best_front_wts = copy.deepcopy(model_frontend.state_dict())
                    best_cnn_wts = copy.deepcopy(model_cnn.state_dict())
                    best_acc = epoch_acc
                countV -= 1
        countT = trainSteps
        countV = valSteps
        # Calculate average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # Calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainLoader.dataset)
        valCorrect = valCorrect / len(valLoader.dataset)

        # Update training history
        hist['train_loss'][e] += avgTrainLoss
        hist['train_acc'][e] += trainCorrect
        hist['val_loss'][e] += avgValLoss
        hist['val_acc'][e] += valCorrect
        hist['t_value'][e] += t_val
        # Confusion matrix
        conf_mat=confusion_matrix(val_cm['pred'].numpy(), val_cm['truth'].numpy())
        print('\n-----Confusion Matrix-----\n')
        print(conf_mat)
        # Calculate F1
        f1 = f1_score(val_cm['pred'].numpy(), val_cm['truth'].numpy(), average='binary')
        hist['F1'][e] += f1
        # Print model training and validation information
        print('\n[INFO] EPOCH: {}/{}'.format(e + 1, epochs))
        print('Train loss: {:.6f}, Train accuracy: {:.4f}'.format(
            avgTrainLoss, trainCorrect))
        print('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(
            avgValLoss, valCorrect))
        print('F1 Score: {:.6f}'.format(f1))

        # Per-class accuracy
        class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
        hist['no_bird_acc'][e] += class_accuracy[0]
        hist['has_bird_acc'][e] += class_accuracy[1]
        print('No bird Accuracy: %.2f'%class_accuracy[0])
        print('Has bird Accuracy: %.2f'%class_accuracy[1])
        epoch_end = time.time()
        hist['Time'].append(epoch_end-epoch_start)
        print('Epoch duration: %.2f sec.\n\n-----------------------------------------------' % (epoch_end - epoch_start))

    # Finish measuring training time
    endTime = time.time()
    print('[INFO] total time taken to train the model: {:.2f}s'.format(
        endTime - startTime))

    # Time log for saved figs
    logger = datetime.now()
    logger = logger.strftime("%m_%d_%Y_%H_%M_%S")

    # Dataframe to csv of epoch training/validation data
    results_df = pd.DataFrame.from_dict(hist, orient='columns')
    results_df.to_csv('../results/experiments/pcentr_all/result_data/pcentr_'+val_fold)
    # Figures
    plt.style.use('ggplot')
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,10))

    # Loss
    ax[0].plot(hist['train_loss'], label='Training Loss')
    ax[0].plot(hist['val_loss'], label='Validation Loss')
    ax[0].set_title('Training/Validation Loss: PCEN')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='lower left')

    # Accuracy
    ax[1].plot(hist['train_acc'], label='Training Accuracy')
    ax[1].plot(hist['val_acc'], label='Validation Accuracy')
    ax[1].set_title('Training/Validation Accuracy: PCEN')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('acc')
    ax[1].legend(loc='lower left')

    plt.savefig('../results/experiments/pcentr_all/plots/pcentr_'+str(rd_fold[i])+'.png')

print('--------------DONE----------------')
