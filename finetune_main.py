import argparse
import os

import torch

from finetune import finetune
from logger import get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-log-dir', required=True)
    parser.add_argument('--lr-ft', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, required=True)
    args = parser.parse_args()

    ckpt = torch.load(os.path.join(args.reg_log_dir, 'ckpt.pt'))
    pruner = ckpt['pruner']
    logger = get_logger(args.reg_log_dir, False, 'finetune_log.pt')
    finetune(pruner, args.lr_ft, args.num_epochs, 10, logger, args.reg_log_dir)
