import argparse
import os
from datetime import datetime

import torch

import dataset
from logger import get_logger
from pruner import GRegPrunerI, GraspPruner
from resnet import resnet56
from vgg import vgg19


def rm_parallel_module_name(state_dict):
    ret = {}
    for key, values in state_dict.items():
        if 'module.' in key:
            ret[key.replace('module.', '')] = values
        else:
            ret[key] = values

    return ret


def main_worker(
        arch,
        batch_size,
        num_worker,
        pretrained_model_path,
        block_candidate_path,
        clear_prev_log,
        block_size_mode,
        log_directory,
        reg_mode,
        init_pr_over_kp_threshold,
        init_model_sparsity,
        layer_sparsity_delta,
        sparsity_assignment
):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = get_logger(log_directory, clear_prev_log)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    trainset, testset = dataset.get_dataset()
    trainloader, testloader = dataset.get_dataloader(
        trainset=trainset,
        testset=testset,
        batch_size=batch_size,
        num_worker=num_worker,
        drop_last_batch=False
    )

    if arch == 'vgg19':
        model = vgg19()
    elif arch == 'resnet56':
        model = resnet56()
    else:
        raise NotImplementedError

    if reg_mode == 1:
        logger.info(f"Start regularization pruning")
        ckpt = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(rm_parallel_module_name(ckpt))
        pruner = GRegPrunerI(
            model=model,
            device=device,
            trainloader=trainloader,
            testloader=testloader,
            lr_prune=0.001,
            momentum=0.9,
            weight_decay=0.0005,
            valid_block_pruning_path=block_candidate_path,
            pr_ratio=0.9,
            reg_ceiling=1,
            update_reg_interval=10,
            epsilon_lambda=0.0001,
            logger=logger,
            log_directory=log_directory,
            save_interval=9999,
            block_size_mode=block_size_mode,
            init_pr_over_kp_threshold=init_pr_over_kp_threshold,
            init_model_sparsity=init_model_sparsity,
            layer_sparsity_delta=layer_sparsity_delta,
            sparsity_assignment=sparsity_assignment
        )
    elif reg_mode == 3:
        logger.info(f"Start block grasp pruning")
        pruner = GraspPruner(
            model=model,
            global_ratio=0.75,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            lr_prune=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            valid_block_pruning_path=block_candidate_path,
            block_size_mode=block_size_mode,
            logger=logger,
            log_directory=log_directory
        )
    else:
        raise NotImplementedError

    pruner.prune()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size-mode", type=str, required=True, help='choice={min, max, unstructured}')
    parser.add_argument("--reg-mode", type=int, required=True, help='choice={1,2}')
    parser.add_argument("--sparsity-assignment", type=str, required=True, help='choice={uniform, rf}')
    parser.add_argument("--arch", type=str, required=True, help='choice={vgg19, resnet56}')
    args = parser.parse_args()
    block_size_mode = args.block_size_mode
    reg_mode = args.reg_mode
    sparsity_assignment = args.sparsity_assignment

    block_candidate_path = 'valid_block_search_space/cifar_vgg19bn_sparsednn_tau_acc_20_tau_lat_1.5.pt'
    pretrained_model_path = 'model_checkpoint.pth'

    log_directory = f'Experiments/greg{reg_mode}_logs_{block_size_mode}_{sparsity_assignment}_{datetime.now().strftime("%m%d%Y_%H%M%S")}'

    main_worker(
        batch_size=256,
        num_worker=14,
        pretrained_model_path=pretrained_model_path,
        block_candidate_path=block_candidate_path,
        clear_prev_log=True,
        block_size_mode=block_size_mode,
        log_directory=log_directory,
        reg_mode=reg_mode,
        init_pr_over_kp_threshold=2,
        init_model_sparsity=0.9,
        layer_sparsity_delta=0.0001,
        sparsity_assignment=sparsity_assignment,
        arch=args.arch
    )
