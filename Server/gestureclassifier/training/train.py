import os
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log                  # Logging
import copy
import cv2

from model_update import create_model
from tqdm import tqdm
from datetime import timedelta


# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    help='dataset to train on: ' + ' | '.join(DataFactory.dataset_names),
                    default='uvrhand') # sbu   uvrhand
parser.add_argument('--seed', type=int, metavar='N',
                    help='random number generator seed, use "-1" for random seed',
                    default=-1)
parser.add_argument('--num-synth', type=int, metavar='N',
                    help='number of synthetic samples to generate',
                    default=8)
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)
parser.add_argument('--model', type=int, metavar='N',
                    help='number of model version',
                    default=1)      # 0 : original, 1: modified
parser.add_argument('--name', type=str, metavar='N',
                    default="v4_")      # 0 : original, 1: modified

parser.add_argument('--lr', type=float, metavar='N',
                    default=0.001)
parser.add_argument('--batch', type=int, metavar='N',
                    default=16)
parser.add_argument('--decay', type=float, metavar='N',
                    default=0.001)
parser.add_argument('--epoch', type=int, metavar='N',
                    default=100)
parser.add_argument('--patience', type=int, metavar='N',
                    help='early stopping patience',
                    default=5)
parser.add_argument('--min-delta', type=float, metavar='N',
                    help='minimum improvement for early stopping',
                    default=0.01)

# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
seed = int(time.time()) if args.seed == -1 else args.seed
use_cuda = torch.cuda.is_available() and args.use_cuda

eval_freq = 5
save_freq = 5

save_path = ""

# ----------------------------------------------------------------------------------------------------------------------
def main():
    global save_path
    # Load the dataset
    log.set_dataset_name(args.dataset)
    dataset = DataFactory.instantiate(args.dataset, args.num_synth)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)

    save_path = f"./checkpoints/{seed}_{args.name}"
    os.makedirs(save_path, exist_ok=True)

    # Run each fold and average the results
    accuracies = []

    try:
        for fold_idx in range(dataset.num_folds):
            start = time.time()
            log('Running fold "{}"...'.format(dataset.folds[fold_idx]))

            test_accuracy = run_fold(dataset, fold_idx, use_cuda)
            accuracies += [test_accuracy]

            log('Fold "{}" complete, final accuracy: {}'.format(dataset.folds[fold_idx], test_accuracy))
            end = time.time()

            elapsed = end - start
            remain = elapsed * (dataset.num_folds - 1 - fold_idx)
            formatted_time = str(timedelta(seconds=remain))
            print("남은 시간:", formatted_time)

    except KeyboardInterrupt:
        if use_cuda:
            torch.cuda.empty_cache()

    log('')
    log('-----------------------------------------------------------------------')
    log('Training complete!')
    log('Average accuracy: {}'.format(np.mean(accuracies)))
    log('Each accuracy: {}'.format(accuracies))


# ----------------------------------------------------------------------------------------------------------------------
def run_fold(dataset, fold_idx, use_cuda):
    """
    Trains/tests the model on the given fold
    """
    global save_path

    # Instantiate the model, loss measure and optimizer
    if args.model == 0:
        print("creating original model")
        model = DeepGRU(dataset.num_features, dataset.num_classes)
    elif args.model == 1:
        print("creating modified model")
        model = create_model(num_features=dataset.num_features, num_classes=dataset.num_classes)

    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.decay)
    # optimizer = torch.optim.AdamW([
    #     {'params': model.gcn.parameters(), 'lr': 1e-4},
    #     {'params': model.gru1.parameters(), 'lr': 5e-4},
    #     {'params': model.classifier.parameters(), 'lr': 1e-3}
    # ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Create data loaders
    train_loader, test_loader = dataset.get_data_loaders(fold_idx,
                                                         args.batch,
                                                         shuffle=True,
                                                         random_seed=seed+fold_idx)

    best_train_accuracy = 0
    best_test_accuracy = 0

    early_stopping = EarlyStopping(patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=True
    )

    # Train the model
    for epoch in range(args.epoch):
        loss_meter = AverageMeter()
        train_meter = AverageMeter()
        test_meter = AverageMeter()

        #
        # Training loop
        #
        model.train()

        for batch in tqdm(train_loader, desc="Loading training batches"):

            optimizer.zero_grad()

            accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update stats
            loss_meter.update(loss.item(), curr_batch_size)
            train_meter.update(accuracy, curr_batch_size)

        train_accuracy = train_meter.avg

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        log('Epoch: [{0}]    lr : {1:.8f}'.format(epoch, optimizer.param_groups[0]['lr']))
        log('       [Avg Loss]          {loss.avg:.6f}'.format(loss=loss_meter))
        log('       [Training]   Prec@1 {top1.avg:.6f} Max {max:.6f}'
             .format(top1=train_meter, max=best_train_accuracy))

        scheduler.step()

        #
        # Testing loop
        #
        if epoch % eval_freq == 0:

            model.eval()

            with torch.no_grad():
                test_loss_meter = AverageMeter()

                for batch in tqdm(test_loader, desc="Loading test batches"):

                    accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)
                    test_loss_meter.update(loss.item(), curr_batch_size)
                    test_meter.update(accuracy, curr_batch_size)

                test_accuracy = test_meter.avg

                # Update best accuracies
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy

                log('       [Avg Loss]          {loss.avg:.6f}'.format(loss=test_loss_meter))
                log('       [Validation] Prec@1 {top1:.6f} Max {max:.6f}'
                     .format(top1=test_accuracy, max=best_test_accuracy))

                if early_stopping(test_accuracy, model):
                    log(f'Early stopping triggered at epoch {epoch}')
                    log(f'Best validation accuracy: {early_stopping.best_score:.6f}')
                    break

            if loss_meter.avg <= 1e-6 or best_test_accuracy == 100:
                break

        if epoch % save_freq == 0:
            os.makedirs(save_path + f"/{dataset.folds[fold_idx]}", exist_ok=True)
            save_fpath = save_path + f"/{dataset.folds[fold_idx]}/checkpoint-{epoch}.tar"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_fpath)

    os.makedirs(save_path + f"/{dataset.folds[fold_idx]}", exist_ok=True)
    save_fpath = save_path + f"/{dataset.folds[fold_idx]}/checkpoint.tar"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_fpath)

    return best_test_accuracy


# ----------------------------------------------------------------------------------------------------------------------
def run_batch(batch, model, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, labels = batch
    # examples uvrhand : 64 12 78

    if use_cuda:
        examples = examples.cuda()
        labels = labels.cuda()

    # Forward and loss computation
    if args.model == 0:
        outputs = model(examples)
    elif args.model == 1:
        outputs = model(examples)

    loss = criterion(outputs, labels)

    # Compute the accuracy
    predicted = outputs.argmax(1)
    correct = (predicted == labels).sum().item()
    curr_batch_size = labels.size(0)
    accuracy = correct / curr_batch_size * 100.0

    return accuracy, curr_batch_size, loss


# ----------------------------------------------------------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_state)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_model_state = copy.deepcopy(model.state_dict())


if __name__ == '__main__':
    main()
