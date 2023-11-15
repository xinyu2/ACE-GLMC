import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
from torchvision.ops import box_convert
import torch.nn.functional as F
from utils import util
from utils.util import *
import datetime
import math
from sklearn.metrics import confusion_matrix
import warnings

class Trainer(object):
    def __init__(self, args, model=None,train_loader=None, val_loader=None,weighted_train_loader=None,per_class_num=[],log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.label_weighting = args.label_weighting
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr,weight_decay=args.weight_decay)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.log = log
        self.beta = args.beta
        self.lambdas = [args.L1, args.L2, args.L3]
        self.f0 = args.f0
        self.mask = self.get_mask()
        print(f"mask={self.mask}")
        self.update_weight()

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    def get_mask(self):
        mask = torch.eye(self.num_classes, dtype=torch.float32, requires_grad=False).cuda()
        for k in range(self.num_classes):
            if self.cls_num_list[k]>=100:
                mask[k,k] = self.lambdas[0]
            elif self.cls_num_list[k]>=20:
                mask[k,k] = self.lambdas[1]
            else:
                mask[k,k] = self.lambdas[2]
        return mask
            
    def train(self, optim_state):
        best_acc1 = 0
        # ========================
        # resume optimizer.lr 
        # ========================
        if optim_state is not None:
            self.optimizer.load_state_dict(optim_state)
        elif self.args.resume is not None:
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, self.args.start_epoch - 1, self.args)
            else:
                for _ in range(self.args.start_epoch - 1):
                    self.train_scheduler.step()

        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1), 1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # Data augmentation
                lam = np.random.beta(self.beta, self.beta)

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1, org2=input_org_2,
                                                                                        invs1=input_invs_1,
                                                                                        invs2=input_invs_2,
                                                                                        label_org=one_hot_org,
                                                                                        label_invs=one_hot_invs,
                                                                                        label_org_w=one_hot_org_w,
                                                                                        label_invs_w=one_hot_invs_w)


                output_1, output_cb_1, z1, p1 = self.model(mix_x, train=True)
                output_2, output_cb_2, z2, p2 = self.model(cut_x, train=True)
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)
                if self.args.lossfn == 'ori':
                    loss_mix = self.mixupLoss(output_1, mixup_y)
                    loss_cut = self.mixupLoss(output_2, mixcut_y)
                    loss_mix_w = self.mixupLoss(output_cb_1, mixup_y_w)
                    loss_cut_w = self.mixupLoss(output_cb_2, cutmix_y_w)
                elif self.args.lossfn == 'ace':
                    loss_mix = self.mixupACE1(output_1, mixup_y)
                    loss_cut = self.mixupACE1(output_2, mixcut_y)
                    loss_mix_w = self.mixupACE1(output_cb_1, mixup_y_w)
                    loss_cut_w = self.mixupACE1(output_cb_2, cutmix_y_w)

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (1 - alpha) * rebalance_loss + self.contrast_weight * contrastive_loss

                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
                    # evaluate on validation set
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

    def train_box(self, optim_state):
        best_acc1 = 0
        # ========================
        # resume optimizer.lr 
        # ========================
        if optim_state is not None:
            self.optimizer.load_state_dict(optim_state)
        elif self.args.resume is not None:
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, self.args.start_epoch - 1, self.args)
            else:
                for _ in range(self.args.start_epoch - 1):
                    self.train_scheduler.step()

        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets, _) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs, boxes = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs, box_inv = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]
                box_inv = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1), 1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                # print(f"org1={input_org_1.shape}, {input_org_1.size()[0]}, inv={input_invs[0].shape}, {input_invs_1.shape}," \
                #        "one_hot_org_w={one_hot_org_w.shape}, one_hot_invs_w={one_hot_invs_w.shape}")
                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # Data augmentation
                lam = np.random.beta(self.beta, self.beta)

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed_box(org1=input_org_1, org2=input_org_2,
                                                                                        invs1=input_invs_1,
                                                                                        invs2=input_invs_2,
                                                                                        invbox=box_inv,
                                                                                        label_org=one_hot_org,
                                                                                        label_invs=one_hot_invs,
                                                                                        label_org_w=one_hot_org_w,
                                                                                        label_invs_w=one_hot_invs_w)


                output_1, output_cb_1, z1, p1 = self.model(mix_x, train=True)
                output_2, output_cb_2, z2, p2 = self.model(cut_x, train=True)
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)
                # if i == 0:
                #     print(f"{len(inputs)}, org1={input_org_1.shape}, org2={input_org_2.shape}")
                #     print(f"mixup-y={mixup_y}, mixcut-y={mixcut_y}, mixup-y-w={mixup_y_w}, cutmix-y-w={cutmix_y_w}")
                # loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                # loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                # loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                # loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))
                if self.args.lossfn == 'ori':
                    loss_mix = self.mixupLoss(output_1, mixup_y)
                    loss_cut = self.mixupLoss(output_2, mixcut_y)
                    loss_mix_w = self.mixupLoss(output_cb_1, mixup_y_w)
                    loss_cut_w = self.mixupLoss(output_cb_2, cutmix_y_w)
                elif self.args.lossfn == 'ace':
                    loss_mix = self.mixupACE1(output_1, mixup_y)
                    loss_cut = self.mixupACE1(output_2, mixcut_y)
                    loss_mix_w = self.mixupACE1(output_cb_1, mixup_y_w)
                    loss_cut_w = self.mixupACE1(output_cb_2, cutmix_y_w)

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (1 - alpha) * rebalance_loss + self.contrast_weight * contrastive_loss

                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
                    # evaluate on validation set
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)
                    
    def validate(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = self.model(input, train=False)

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1 , flag='val', top1=top1, top5=top5))

            self.log.info(output)
            out_cls_acc = '%s Class Accuracy: %s' % (
            'val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))

            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg

    def SimSiamLoss(self,p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def mixupLoss(self,mix_output, mix_y):  # negative cosine similarity
        yfx = F.log_softmax(mix_output, dim=1) * mix_y 
        s = torch.sum(yfx, dim=1)
        l = -torch.mean(s)
        return l

    def mixupACE1(self,mix_output, mix_y):  # negative cosine similarity
        ylfx = F.log_softmax(mix_output, dim=1) * mix_y # Bs x K
        yfx = F.softmax(mix_output, dim=1) * mix_y # Bs x K
        f0 = torch.zeros_like(mix_y, requires_grad=False).cuda() # Bs x K
        f0[mix_y!=0] = self.f0 # Bs x K 
        f = f0 - yfx # Bs x K 
        f[f<0] = 0.0 # hinge f
        one = torch.zeros_like(mix_y, requires_grad=False).cuda() # Bs x K
        one[mix_y!=0] = 1.0 # Bs x K : two-hot
        lam = torch.matmul(one, self.mask) # Bs x K : lambda
        lambdaf = one + lam * f # Bs x K: 1 + lambda dot (f0-fx)
        ace1 = torch.multiply(ylfx, lambdaf)
        s = torch.sum(ace1, dim=1)
        l = -torch.mean(s)
        return l    
                    
    def paco_adjust_learning_rate(self,optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(f"{epoch} lr = {lr}")    



