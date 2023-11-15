import shutil
from torch.utils import data
import copy
import os
from imbalance_data.cifar100Imbanlance import *
from imbalance_data.cifar10Imbanlance import *
from imbalance_data.dataset_lt_data import *
import utils.moco_loader as moco_loader
from utils.randaugment import rand_augment_transform
import numpy as np

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def guided_bbox(size, box):
    W = size[2]
    H = size[3]
    cut_w = np.ceil(W * (box[2]-box[0])).astype(int)
    cut_h = np.ceil(H * (box[3]-box[1])).astype(int)
    # uniform
    cx = np.random.randint(W - np.ceil(cut_w/2).astype(int))
    cy = np.random.randint(H - np.ceil(cut_h/2).astype(int))

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h

    x1 = int(W * box[0])
    y1 = int(H * box[1])
    x2 = int(W * box[2])
    y2 = int(H * box[3])
    if (bbx2 - bbx1) > (x2 - x1):
        x2 += 1
    elif (bbx2 - bbx1) < (x2 - x1):
        x2 -= 1
    if (bby2 - bby1) > (y2 - y1):
        y2 += 1
    elif (bby2 - bby1) < (y2 - y1):
        y2 -= 1
    return (bbx1, bby1, bbx2, bby2), (x1, y1, x2, y2)

def get_transform(dataset):
    if dataset == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform_train,transform_val

    if dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform_train, transform_val

    if dataset == "ImageNet-LT":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

    if dataset == "iNaturelist2018":
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best, epoch):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if epoch % 20 == 0:
        filename = '%s/%s/%s_ckpt.pth.tar' % (args.root_model, args.store_name, str(epoch))
        torch.save(state, filename)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def GLMC_mixed(org1, org2, invs1, invs2, label_org, label_invs, label_org_w, label_invs_w, alpha=1):
    lam = np.random.beta(alpha, alpha)

    # mixup
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    # cutmix
    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)
    org2[:, :, bbx1:bbx2, bby1:bby2] = invs2[:, :, bbx1:bbx2, bby1:bby2]

    lam_cutmix = lam
    cutmix_y = lam_cutmix * label_org + (1 - lam_cutmix) * label_invs
    cutmix_y_w = lam_cutmix * label_org_w + (1 - lam_cutmix) * label_invs_w

    return mixup_x, org2, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w

def GLMC_mixed_box(org1, org2, invs1, invs2, invbox, label_org, label_invs, label_org_w, label_invs_w, alpha=1):
    lam = np.random.beta(alpha, alpha)

    # mixup
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    # cutmix or guided-mix
    cutmix_y = torch.zeros_like(label_org)
    cutmix_y_w = torch.zeros_like(label_org_w)
    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)
    lam_cutmix = lam
    for i, box in enumerate(invbox):
        if np.all(box > 0):
            box = np.clip(box, 0.0, 1.0)
            lam_cutmix = 1 - (box[2] - box[0]) * (box[3] - box[1])
            if lam_cutmix > 0.5: 
                (bbx1, bby1, bbx2, bby2), (x1, y1, x2, y2) = guided_bbox(org2.size(), box)
                # print(f" box={box}, ori: {bbx1, bby1, bbx2, bby2}, inv:{x1, y1, x2, y2}")
                org2[i, :, bbx1:bbx2, bby1:bby2] = invs2[i, :, x1:x2, y1:y2]
            else:
                lam_cutmix = lam
                org2[i, :, bbx1:bbx2, bby1:bby2] = invs2[i, :, bbx1:bbx2, bby1:bby2]    
        else:    
            org2[i, :, bbx1:bbx2, bby1:bby2] = invs2[i, :, bbx1:bbx2, bby1:bby2]

        cutmix_y[i] = lam_cutmix * label_org[i] + (1 - lam_cutmix) * label_invs[i]
        cutmix_y_w[i] = lam_cutmix * label_org_w[i] + (1 - lam_cutmix) * label_invs_w[i]

    return mixup_x, org2, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w

def print_model_param_nums(model=None):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total