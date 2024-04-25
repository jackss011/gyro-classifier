import argparse
import time
import six
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from pytz import timezone
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from delta_regimes import *

def write_num_to_file(num, filename):
    with open(filename, 'w') as f:
        f.write('%s\n' % str(num))

def write_nums_to_file(filename, *numbers):
    with open(filename, 'w') as f:
        for num in numbers:
            f.write(str(num) + '\n')

def write_args_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in six.iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def choose_modify_delta(delta_regime):
    delta_functions = {
        "lin": delta_linear,
        "exp": delta_exp,
        "sqrt": delta_sqrt,
        "square": delta_square,
        "log": delta_mult_log,
        "const": delta_0, # per tenere delta costante
    }
    
    return delta_functions.get(delta_regime, delta_0) # restituisci da dizionario, no key restituisce delta_0


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet_ternary', choices=model_names)
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--delta', default=0.0, type=float,
                    help='delta interval for ternary conversion of weights')
parser.add_argument('--maxdelta', default=1.0, type=float,
                    help='max value of delta interval')
parser.add_argument('--delta_regime', default="", type=str,
                    help='regime of delta increase (default:"", examples: "lin", "log", "exp", "sqrt", "square")')
parser.add_argument('--multiplier', default=1.0, type=float,
                    help='delta increment multiplier')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    mod_delta = choose_modify_delta(args.delta_regime)

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = datetime.now(timezone('Europe/Rome')).strftime('%Y%m%d%H%M%S')
    model_name_path = args.model + '_' + args.save
    if args.model == 'resnet_ternary':
        model_name_path = 'delta' + str(args.delta) + '_' + str(args.delta_regime) + '_max' + str(args.maxdelta) + '_m' + str(args.multiplier) + '_e' + str(args.epochs) + '_' + model_name_path
    save_path = os.path.join(args.results_dir, model_name_path)
    print("il path è: ", save_path)

    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    write_args_to_file(args, save_path + '/arguments.txt')
    log_writer = SummaryWriter(log_dir=os.path.join('./logs/', model_name_path))

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}
    # model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'delta': args.delta}

    if args.delta != '':
        model_config = dict(model_config, **{'delta': args.delta, 'multiplier': args.multiplier})

    if args.model_config != '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):                                   
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'f')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code                                     
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer, 'lr': args.lr, 'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'], download=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'], download=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)

    # Save histogram (uncomment below to create path for hist)
    #save_path_histogram = save_path + '/histogram'
    #if not os.path.exists(save_path_histogram):
    #    os.mkdir(save_path_histogram)
    '''
    # plot first histogram
    plt.hist(model.cpu().return_weights_org(), histtype='step', bins=100)
    plt.savefig(save_path_histogram + '/hist00.png')
    plt.close()
    model.cuda()
    '''

    for epoch in range(args.start_epoch, args.epochs):

        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, epoch)

        # count number of weights set to zero, ones, minus ones and entropy
        n_zeroes_perc = model.count_zero_weights() / num_parameters * 100
        n_ones_perc = model.count_one_weights() / num_parameters * 100
        n_minus_ones_perc = model.count_minus_one_weights() / num_parameters * 100
        probabilities = torch.tensor([n_zeroes_perc / 100, n_ones_perc / 100, n_minus_ones_perc / 100])
        entropy  = -torch.sum(probabilities * torch.log2(probabilities + 1e-9)).item()
        print('% of parameters set to 0: ', str(n_zeroes_perc))

        # plot histograms, uncomment below to save histograms every 10 epochs and below 50 epoch
        ''' 
        if (epoch + 1) % 10 == 0 or (epoch + 1) < 50:
            plt.hist(model.cpu().return_weights_org(), histtype='step', bins=100)
            plt.savefig(save_path_histogram + '/hist_epoch_' + str(epoch + 1) + '_org.png')
            plt.close()
            plt.hist(model.cpu().return_weights(), histtype='step', bins=100)
            plt.savefig(save_path_histogram + '/hist_epoch_' + str(epoch + 1) + '_tri.png')
            plt.close()
            model.cuda()
        '''

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        # write_num_to_file(best_prec1, save_path + '/best_te_acc.txt') # per salvare solo best_te_acc
        if is_best:
            best_epoch = epoch + 1
            best_sparsity = n_zeroes_perc
            best_entropy = entropy
            write_nums_to_file(save_path + '/best_results.txt', best_prec1, best_sparsity, best_epoch, best_entropy) 


        log_writer.add_scalar('train/tr_loss', train_loss, epoch+1)
        log_writer.add_scalar('train/tr_acc', train_prec1, epoch+1)
        log_writer.add_scalar('train/n_zero_params', n_zeroes_perc, epoch+1)
        log_writer.add_scalar('train/delta', model.delta, epoch + 1)
        log_writer.add_scalar('train/absolute mean parameter avg', epoch+1)
        log_writer.add_scalar('test/te_loss', val_loss, epoch+1)
        log_writer.add_scalar('test/te_acc', val_prec1, epoch+1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('Epoch: {0}\t'                             
                     'Training Loss {train_loss:.4f} \t'        
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     'Perc of Params set to 0 {n_zeroes_perc:.4f}'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5, n_zeroes_perc=n_zeroes_perc))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5, 
                    n_zeroes_perc=n_zeroes_perc, n_ones_perc=n_ones_perc, n_minus_ones_perc=n_minus_ones_perc, 
                    entropy=entropy)
        results.save()

        # update delta
        new_delta = min(args.delta + mod_delta(args.delta, args.multiplier, epoch, args.epochs), args.maxdelta)        

        for module in model.modules():
            if not isinstance(module, nn.Sequential):
                module.delta = new_delta




def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()

        if not training:
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                output = model(input_var)
        else:
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            output = model(input_var)

        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.print_freq == 0:
            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'                       
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'       
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                 epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    model.train()
    return forward(data_loader, model, criterion, epoch, training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    model.eval()
    return forward(data_loader, model, criterion, epoch, training=False, optimizer=None)


if __name__ == '__main__':
    main()
