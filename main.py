import argparse
from train import *
from valid import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='../DIV2K_train_HR', help='path to images')
parser.add_argument('--batchSize', type=int, default=96, help='input batch size')
parser.add_argument('--gpu_ids', type=str, default='2,3,4,5,6,7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--experiment', default=None, help='Where to store models')
parser.add_argument('--fineSize', type=int, default=384, help='then crop to this size')
parser.add_argument('--n_epoch_init', type=int, default=10, help='# of iter at pretrained')
parser.add_argument('--n_epoch', type=int, default=2000, help='# of iter at training')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning rate for pretrain')
parser.add_argument('--lr_decay', type=float, default=0.1, help='initial learning rate for adam')
parser.add_argument('--train', type=bool, default=True, help='train or valid')

opt = parser.parse_args()
print(opt)

if opt.train:
    train(opt)
else:
    valid(opt)
