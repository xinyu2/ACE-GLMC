import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data
import logging
import datetime
import math
from sklearn.metrics import confusion_matrix
from Trainer import Trainer

best_acc1 = 0

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        print("=> creating model '{}'".format('resnext50_32x4d'))
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        return net
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
        return net

def get_dataset(args):
    transform_train,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=args.root)
        print("load cifar10")
        return trainset,testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(args.root,'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
        print("load cifar100")
        return trainset,testset

    if args.dataset == 'ImageNet-LT':
        dir_train_txt = os.path.join(args.root, 'data_txt', 'ImageNet_LT_train.txt')
        dir_test_txt = os.path.join(args.root, 'data_txt', 'ImageNet_LT_test.txt')
        lt_root = os.path.join(args.root, 'ImageNet')
        path_to_box = os.path.join(lt_root, 'imagenet_box.pickle') # GroundDINO boxes 
        trainset = dataset_lt_data.LT_box_Dataset(lt_root, dir_train_txt, util.TwoCropTransform(transform_train), path_to_box)
        testset = dataset_lt_data.LT_Dataset(lt_root, dir_test_txt, transform_val)
        return trainset,testset

    if args.dataset == 'iNaturelist2018':
        dir_train_txt = os.path.join(args.root, 'data_txt', 'iNaturalist18_train.txt')
        dir_test_txt = os.path.join(args.root, 'data_txt', 'iNaturalist18_val.txt')
        lt_root = os.path.join(args.root, 'iNaturalist18')
        trainset = dataset_lt_data.LT_Dataset(lt_root, dir_train_txt,util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(lt_root, dir_test_txt,transform_val)
        return trainset,testset

def main():
    args = parser.parse_args()
    print(args)
    curr_time = datetime.datetime.now()
    lam1 = args.L1
    lam2 = args.L2
    lam3 = args.L3
    f0 = args.f0
    parms = [lam1, lam2, lam3, f0]
    parmstr = "-".join([str(p) for p in parms])
    args.store_name = ''.join([args.dataset, args.arch,str(args.imbanlance_rate), parmstr
            ,datetime.datetime.strftime(curr_time, '%m-%d-%H%M%S')])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    main_worker(args.gpu, args)

def main_worker(gpu, args):

    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    optim_state = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            # model.load_state_dict(checkpoint['state_dict'])
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v
            model.load_state_dict(new_state_dict)
            optim_state = checkpoint.get('optimizer_state_dict', None)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    # Data loading code
    t0 = time.time()
    train_dataset,val_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes
    t1 = time.time()
    print(f">> data load time= {(t1 - t0)} seconds")
    cls_num_list = train_dataset.get_per_class_num()
    train_sampler = None
    drop_last = False
    if args.dataset == 'iNaturelist2018':
        drop_last = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),\
                                               num_workers=args.workers, persistent_workers=True,pin_memory=True, sampler=train_sampler, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,\
                                             num_workers=args.workers, persistent_workers=True,pin_memory=True, drop_last=drop_last)

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)
    train_sampler = None
    weighted_train_loader = None

    #weighted_loader
    cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.workers, \
                                                        persistent_workers=True,pin_memory=True,sampler=weighted_sampler, drop_last=drop_last)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    print("Training started!")
    trainer = Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list,log=logging)
    if args.dataset == 'ImageNet-LT':
        trainer.train_box(optim_state)
    else:    
        trainer.train(optim_state)    
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))

if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    parser.add_argument('--root', type=str, default='./data/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',choices=('resnet18', 'resnet32', 'resnet34', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--beta', type=float, default=0.5, help="augment mixture")
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W',help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')
    parser.add_argument('--resample_weighting', default=0.2, type=float,help='weighted for sampling probability (q(1,k))')
    parser.add_argument('--label_weighting', default=1.0, type=float, help='weighted for Loss')
    parser.add_argument('--contrast_weight', default=10,type=int,help='Mixture Consistency  Weights')
    parser.add_argument('--lossfn', default='ace',type=str,help='loss-function (ori, ace)')
    parser.add_argument('--L1', default=0.0,type=float,help='lambda1-of-ace1')
    parser.add_argument('--L2', default=0.0,type=float,help='lambda2-of-ace1')
    parser.add_argument('--L3', default=0.0,type=float,help='lambda3-of-ace1')
    parser.add_argument('--f0', default=0.0,type=float,help='f0-of-ace1')
    # etc.
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N',help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--root_log', type=str, default='./output/')
    parser.add_argument('--root_model', type=str, default='./output/')
    parser.add_argument('--store_name', type=str, default='./output/')
    main()