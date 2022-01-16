import os

import torch

import dataset
from logger import get_logger
from pruner import GRegPrunerI
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
        batch_size,
        num_worker,
        pretrained_model_path,
        block_candidate_path,
        clear_prev_log
):
    log_directory = 'greg1_pruning_logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = get_logger(log_directory, clear_prev_log)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    ckpt = torch.load(pretrained_model_path, map_location=device)

    model = vgg19()
    model.load_state_dict(rm_parallel_module_name(ckpt))

    trainset, testset = dataset.get_dataset()
    trainloader, testloader = dataset.get_dataloader(
        trainset=trainset,
        testset=testset,
        batch_size=batch_size,
        num_worker=num_worker,
        drop_last_batch=False
    )

    logger.info(f"Start regularization pruning")
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
        stablize_interval=1,
        epsilon_lambda=0.0001,
        logger=logger,
        log_directory=log_directory,
        save_interval=10
    )

    sparse_model = pruner.prune()


if __name__ == '__main__':
    block_candidate_path = 'valid_block_search_space/cifar_vgg19bn_sparsednn_tau_acc_20_tau_lat_1.5.pt'
    pretrained_model_path = 'model_checkpoint.pth'
    main_worker(
        batch_size=256,
        num_worker=8,
        pretrained_model_path=pretrained_model_path,
        block_candidate_path=block_candidate_path,
        clear_prev_log=True
    )
