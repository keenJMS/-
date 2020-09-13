from utils import *
import torch
import argparse
import sys
import torch.backends.cudnn as cudnn
import mydataset_manager
from dataset_loader import ImageDataset
parser = argparse.ArgumentParser(description='Train image model with center loss')
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
def main():
    #config
    config=get_config('/root/proj/JMSreid/config/config.yaml')
    train_epoch=config['train_epoch']

    #gpu_setting
    use_gpu=torch.cuda.is_available()
    pin_memory=True if use_gpu else False
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
    #logs
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    #dataset
    dataset=mydataset_manager.Market1501(root=config['dataset_root'])

    for epoch in range(train_epoch):
        train()
def train():
    pass












if __name__=='__main__':
    main()