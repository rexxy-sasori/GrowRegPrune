import argparse
import os

import torch

from finetune import finetune
from logger import get_logger


def strdict_to_dict(sstr, ttype):
    '''
        '{"1": 0.04, "2": 0.04, "4": 0.03, "5": 0.02, "7": 0.03, }'
    '''
    if not sstr:
        return sstr
    out = {}
    sstr = sstr.strip()
    if sstr.startswith('{') and sstr.endswith('}'):
        sstr = sstr[1:-1]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            k = x.split(':')[0]
            v = ttype(x.split(':')[1].strip())
            out[k] = v
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-log-dir', required=True)
    parser.add_argument('--lr-ft', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, required=True)
    args = parser.parse_args()

    ckpt = torch.load(os.path.join(args.reg_log_dir, 'ckpt.pt'))
    pruner = ckpt['pruner']
    logger = get_logger(args.reg_log_dir, False, 'finetune_log.pt')
    lr_ft = strdict_to_dict(args.lr_ft, float)
    finetune(pruner, lr_ft, args.num_epochs, 10, logger, args.reg_log_dir)
