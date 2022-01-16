import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from nnutils.training_pipeline import accuracy_evaluator
from torch import nn


class PruningArgs:
    def __init__(
            self,
            layer: nn.Module,
            block_dimension: tuple,
            pr_ratio,
            device
    ):
        self.layer = layer
        self.block_dimension = block_dimension
        self.pr_ratio = pr_ratio
        self.is_conv = isinstance(layer, nn.Conv2d)

        self.pr_mask = self._find_target_masks()
        self.kp_mask = 1 - self.pr_mask
        self.reg = torch.zeros_like(layer.weight.data)

        self.pr_mask = self.pr_mask.to(device)
        self.kp_mask = self.kp_mask.to(device)
        self.reg = self.reg.to(device)

    def _find_target_masks(self):
        weight_tensor = self.layer.weight.data

        if self.is_conv:
            return 1 - mask_conv_layer(weight_tensor, self.block_dimension, self.pr_ratio)
        else:
            return 1 - mask_linear_layer(weight_tensor, self.block_dimension, self.pr_ratio)

    def update_reg_single_layer(self, epsilon_lambda):
        self.reg += self.pr_mask * epsilon_lambda * torch.ones_like(self.reg)

    def apply_reg_single_layer(self):
        l2_grad = self.reg * self.layer.weight
        self.layer.weight.grad += l2_grad

    def get_kp_weight_norm(self):
        if self.pr_ratio == 0:
            return torch.norm(self.layer.weight.data) ** 2
        else:
            return torch.norm(self.kp_mask * self.layer.weight.data) ** 2

    def get_pr_weight_norm(self):
        if self.pr_ratio == 0:
            return 0
        else:
            return torch.norm(self.pr_mask * self.layer.weight.data) ** 2

    def finished_updating_reg_single_layer(self, reg_upper_limit):
        return self.reg.max() > reg_upper_limit


class GRegPrunerI:
    def __init__(
            self,
            # model
            model: nn.Module,
            device,

            # data
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,

            # optimizer
            lr_prune,
            momentum,
            weight_decay,

            # pruning block dimension
            valid_block_pruning_path,
            pr_ratio,

            # regI parameter
            reg_ceiling,
            update_reg_interval,
            stablize_interval,
            epsilon_lambda,

            logger,

            # loss function
            criterion=nn.CrossEntropyLoss(),

            # bookkeeping
            log_directory='./greg1_pruning_logs',
            save_interval=1
    ):
        self.logger = logger

        self.device = device
        self.model = model.to(device)

        self.model_orig_state = deepcopy(model.state_dict())
        self.total_param = sum([p.numel() for p in self.model_orig_state.values() if p.dim() == 4 or p.dim() == 2])

        self.trainloader = trainloader
        self.testloader = testloader

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr_prune,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.criterion = criterion

        self.reg_ceiling = reg_ceiling
        self.update_reg_interval = update_reg_interval
        self.stablize_interval = stablize_interval
        self.epsilon_lambda = epsilon_lambda

        self.valid_block_dims = torch.load(valid_block_pruning_path)
        self.pr_ratio = pr_ratio

        self.pr_over_kp_weight_norm_history = []
        self.target_layers, self.target_model_sparsity = self._register_layers()

        self.log_directory = log_directory
        self.save_interval = save_interval

    def _register_layers(self):
        self.logger.info(f"Looking for layers to prune")
        target_layers = {}
        expected_model_sparsity = 0

        count = 0
        pr_over_kp_weight_norm = {}
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if not is_compute_layer(module):
                continue

            block_dims_candidate = self.valid_block_dims[name]
            block_sizes = np.array([v[0] * v[1] for v in block_dims_candidate if v != (1, 1)])
            block_dimension = block_dims_candidate[np.argmin(block_sizes)] if len(block_sizes) >= 1 else (1, 1)
            # block_dimension = (1,1)
            if count == 0 or name == 'classifier':
                pr_ratio = 0
            else:
                pr_ratio = self.pr_ratio

            target_layers[name] = PruningArgs(module, block_dimension, pr_ratio, self.device)

            kp_weight = target_layers[name].get_kp_weight_norm()
            pr_weight = target_layers[name].get_pr_weight_norm()
            pr_over_kp = pr_weight / kp_weight
            self.logger.info(f"Register layer name: {name}, "
                             f"shape: {module.weight.data.shape} "
                             f"using pr ratio {pr_ratio} "
                             f"block dims: {block_dimension} "
                             f"PrWeightNorm/KpWeightNorm={pr_over_kp}")
            pr_over_kp_weight_norm[name] = pr_over_kp
            count += 1
            expected_model_sparsity += module.weight.data.numel() / self.total_param * pr_ratio

        # torch.save(pr_over_kp_weight_norm, f'block_extra_factor/{self.pr_ratio}.pt')
        self.pr_over_kp_weight_norm_history.append(pr_over_kp_weight_norm)
        self.logger.info(f"Expected model sparsity: {expected_model_sparsity}")
        return target_layers, expected_model_sparsity

    def prune(self):
        self.model = self.model.train()
        num_iter = 0
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0

        while not self.finish_updating_reg_all_layer():
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.model.train()
                y_ = self.model(inputs)

                if num_iter % self.update_reg_interval == 0:
                    self.update_reg()

                loss = self.criterion(y_, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.apply_reg()
                self.optimizer.step()

                train_acc, train_loss = accuracy_evaluator.eval(
                    self.model, self.device, self.trainloader, self.criterion, print_acc=False
                )

                self.logger.info("TRAIN Acc1 = %.4f, Iter = %d @ Batch [%d]/[%d]" % (
                    train_acc, num_iter, batch_idx + 1, len(self.trainloader)))

                pr_over_kp_weight_norm = {}
                self.logger.info(f"Evaluating PrWeightNorm/KpWeightNorm @ Iter = {num_iter}")
                for name, layer_args in self.target_layers.items():
                    kp_weight = layer_args.get_kp_weight_norm()
                    pr_weight = layer_args.get_pr_weight_norm()
                    pr_over_kp = pr_weight / kp_weight
                    pr_over_kp_weight_norm[name] = pr_over_kp
                    self.logger.info(f"Name: {name}, Pr/Kp={pr_over_kp}")
                self.pr_over_kp_weight_norm_history.append(pr_over_kp_weight_norm)

                if num_iter % self.save_interval == 0:
                    test_acc, test_loss = accuracy_evaluator.eval(
                        self.model, self.device, self.testloader, self.criterion, print_acc=False
                    )

                    self.logger.info("TEST Acc1 = %.4f, Iter = %d " % (test_acc, num_iter))
                    torch.save(
                        self.pr_over_kp_weight_norm_history,
                        os.path.join(self.log_directory, 'pr_over_kp_weight_norm.pt')
                    )

                    torch.save(
                        {
                            'state_dict': self.model.state_dict(),
                            'test_acc': test_acc,
                            'test_loss': test_loss,
                            'iteration': num_iter,
                            'pruner': self
                        },
                        os.path.join(self.log_directory, 'ckpt.pt')
                    )

                num_iter += 1

        self.logger.info(f'finished updating model while enforcing regularization'
                         f'Last saw test acc: {test_acc}')
        self.logger.info(f'PrWeightNorm/KpWeightNorm before and after imposing regularization')
        for name in self.target_layers.keys():
            self.logger.info(
                f'{name}: {self.pr_over_kp_weight_norm_history[0][name]} (before)'
                f'vs. {self.pr_over_kp_weight_norm_history[-1][name]} (after)'
            )

        self.logger.info('Proceed to stabling model')
        return deepcopy(self.model)

    def update_reg(self):
        for name, target in self.target_layers.items():
            target.update_reg_single_layer(self.epsilon_lambda)

    def apply_reg(self):
        for name, target in self.target_layers.items():
            target.apply_reg_single_layer()

    def finish_updating_reg_all_layer(self):
        for name, target in self.target_layers.items():
            if not target.finished_updating_reg_single_layer(self.reg_ceiling):
                return False

        return True


def is_compute_layer(module):
    return isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)


def matrix2blocks(tensor, nrows_per_block, ncols_per_block):
    tensor_nrows, tensor_ncols = tensor.shape
    ret = tensor.reshape(tensor_nrows // nrows_per_block, nrows_per_block, -1, ncols_per_block)
    ret = torch.transpose(ret, 1, 2)
    ret = ret.reshape(-1, nrows_per_block, ncols_per_block)
    return ret, tensor_nrows // nrows_per_block, tensor_ncols // ncols_per_block


def blocks2matrix(blocks, num_blocks_row, num_blocks_col, nrows_per_block, ncols_per_block):
    ret = blocks.reshape(num_blocks_row, num_blocks_col, nrows_per_block, ncols_per_block)
    ret = torch.transpose(ret, 1, 2)
    ret = ret.reshape(num_blocks_row * nrows_per_block, num_blocks_col * ncols_per_block)
    return ret


def mask_linear_layer(weight, block_dims, alpha):
    br, bc = block_dims
    alpha = alpha

    blocks, num_blocks_row, num_blocks_col = matrix2blocks(weight, br, bc)

    score = torch.norm(blocks, dim=(-2, -1)) ** 2
    _, sorted_indices = torch.sort(score)  # compute sensor of each blocks

    mask = torch.ones_like(blocks)
    mask[sorted_indices[0:int(num_blocks_row * num_blocks_col * alpha)], :, :] = 0
    mask = blocks2matrix(mask, num_blocks_row, num_blocks_col, br, bc)
    return mask


def mask_conv_layer(weight, block_dims, alpha):
    cout, cin, hk, wk = weight.shape
    unroll_weight = weight.reshape(cout, cin * hk * wk)
    unroll_mask = mask_linear_layer(unroll_weight, block_dims, alpha)
    return unroll_mask.reshape(cout, cin, hk, wk)
