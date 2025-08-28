import os
import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from model import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log                  # Logging



# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    help='dataset to train on: ' + ' | '.join(DataFactory.dataset_names),
                    default='uvrhand') # sbu   uvrhand
parser.add_argument('--ckpt', type=str, default="")
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)


# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
seed = int(time.time()) if args.seed == -1 else args.seed
use_cuda = torch.cuda.is_available() and args.use_cuda

eval_freq = 5
save_freq = 5

ckpt_path = f"./checkpoints/{args.ckpt}"


# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Load the dataset
    log.set_dataset_name(args.dataset)
    dataset = DataFactory.instantiate(args.dataset, args.num_synth)

    criterion = nn.CrossEntropyLoss()
    # Instantiate the model, loss measure and optimizer
    model = DeepGRU(dataset.num_features, dataset.num_classes)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Create data loaders
    train_loader, test_loader = dataset.get_data_loaders(0,
                                                         shuffle=False,
                                                         random_seed=seed+0,
                                                         normalize=False)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)
        log('[Validation] Prec@1 {top1:.6f}'.format(top1=accuracy))

    # visualize?



# ----------------------------------------------------------------------------------------------------------------------
def run_batch(batch, model, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch
    # examples uvrhand : 64 160 78

    if use_cuda:
        examples = examples.cuda()
        labels = labels.cuda()

    # Forward and loss computation
    outputs = model(examples, lengths)
    loss = criterion(outputs, labels)

    # Compute the accuracy
    predicted = outputs.argmax(1)
    correct = (predicted == labels).sum().item()
    curr_batch_size = labels.size(0)
    accuracy = correct / curr_batch_size * 100.0

    return accuracy, curr_batch_size, loss


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
