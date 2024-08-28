import argparse
import time
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataloading import loadX, loadY
import torch.optim
import torch.utils.data
import models
from prox_utils import *
from datetime import datetime
from prox_reg import *
from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--gpus', default='0', help='gpus used for training - e.g 0,1,3')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--reg_rate', default=5e-5, type=float, help='Regularization rate')
parser.add_argument('--projection_mode', default='prox', type=str, help='Projection mode')
parser.add_argument('--freeze_epoch', default=-1, type=int, help='Epoch to freeze quantization')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPT', help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=40, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',  help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE', help='evaluate model FILE on validation set')


def main():
    global args, best_prec1, best_prec1_bin
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html') # QUI
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    log_writer = SummaryWriter(log_dir=os.path.join('./logs/', args.save))

    dataset_folder = os.path.join('..', 'dataset', 'dataset1')
    Xtrain = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
    Ytrain = loadY(os.path.join(dataset_folder, "train"), "train")
    XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
    Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")
    numClasses = max(Ytrain)

    trainData = list()
    for i in range(len(Xtrain)):
        sample = [Xtrain[i]]
        trainData.append((torch.tensor(sample, dtype=torch.float32), Ytrain[i] - 1))

    # Create the tensor for testing
    testData = list()
    for i in range(len(XtestRaw1)):
        sample = [XtestRaw1[i]]
        testData.append((torch.tensor(sample, dtype=torch.float32), Ytest1[i] - 1))

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False)

    # create model
    model = models.CNN(numClasses).to(device)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)", args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)", checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([ll.nelement() for ll in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.evaluate:
        validate(testLoader, model, criterion, 0)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bin_op = BinOp(model, if_binary=if_binary)

    # Optionally freeze before training
    if args.freeze_epoch == 1:
        bin_op.quantize(mode='binary_freeze')
        args.projection_mode = None

    # training loop
    best_prec1 = 0
    best_prec1_bin = 0
    try:
        for epoch in range(args.start_epoch, args.epochs):

            br = args.reg_rate * (epoch + 1)
                
            train_loss, train_prec1, train_prec5 = train(trainLoader, model, criterion, epoch, optimizer, br=br,
                                                         bin_op=bin_op, projection_mode=args.projection_mode,
                                                         binarize=False)

            val_loss, val_prec1, val_prec5 = validate(testLoader, model, criterion, epoch, br=br, bin_op=bin_op,
                                                      projection_mode=args.projection_mode, binarize=False)

            val_loss_bin, val_prec1_bin, val_prec5_bin = validate(testLoader, model, criterion, epoch, br=br,
                                                                  bin_op=bin_op, projection_mode=args.projection_mode,
                                                                  binarize=True)

            best_prec1 = max(val_prec1, best_prec1)
            best_prec1_bin = max(val_prec1_bin, best_prec1)
            is_best = best_prec1_bin

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1,
                             'best_prec1_bin': best_prec1_bin,
                             }, is_best, path=save_path)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Training Prec@5 {train_prec5:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1,
                                 train_prec5=train_prec5, val_prec5=val_prec5))
            
            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, val_loss_bin=val_loss_bin,
                        train_error1=train_prec1, val_error1=val_prec1, val_prec1_bin=val_prec1_bin,
                        train_error5=train_prec5, val_error5=val_prec5, val_prec5_bin=val_prec5_bin)
            results.save()

            log_writer.add_scalar('train_loss', train_loss, epoch + 1)
            log_writer.add_scalar('test_loss', val_loss, epoch + 1)
            log_writer.add_scalar('test_loss_bin', val_loss_bin, epoch + 1)
            log_writer.add_scalar('train_acc1', train_prec1, epoch + 1)
            log_writer.add_scalar('train_acc5', train_prec5, epoch + 1)
            log_writer.add_scalar('test_acc1', val_prec1, epoch + 1)
            log_writer.add_scalar('test_acc5', val_prec5, epoch + 1)
            log_writer.add_scalar('test_acc1_bin', val_prec1_bin, epoch + 1)
            log_writer.add_scalar('test_acc5_bin', val_prec5_bin, epoch + 1)

            # Optionally freeze the binarization at a given epoch
            if 0 < args.freeze_epoch == epoch+1:
                bin_op.quantize(mode='binary_freeze')
                args.projection_mode = None

            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None,
            br=0.0, bin_op=None, projection_mode=None, binarize=False):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not training:
        bin_op.save_params()
        if binarize:
            bin_op.quantize('deterministic')
    
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            inputs = inputs.cuda()
            target = target.cuda()

        input_var = Variable(inputs)
        target_var = Variable(target)

        # compute output
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

            # copy parameters according to quantization modes
            if projection_mode == 'prox':
                optimizer.step()
                bin_op.prox_operator(br, reg_type='binary')
                bin_op.clip()
            else:
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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
        del output, loss
        torch.cuda.empty_cache()

    if not training:
        bin_op.restore()
    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, br=0.0, bin_op=None, projection_mode=None, binarize=False):
    model.train()
    return forward(data_loader, model, criterion, epoch, training=True, optimizer=optimizer,
                   br=br, bin_op=bin_op, projection_mode=projection_mode, binarize=binarize)


def validate(data_loader, model, criterion, epoch, br=0.0, bin_op=None, projection_mode=None, binarize=False):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch, training=False, optimizer=None, br=br,
                   bin_op=bin_op, projection_mode=projection_mode, binarize=binarize)


if __name__ == '__main__':
    main()
