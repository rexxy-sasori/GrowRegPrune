import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from nnutils.training_pipeline import accuracy_evaluator
from torch import nn


class GRegIArgs:
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

        self.pr_mask = self._find_init_pruning_masks()
        self.kp_mask = 1 - self.pr_mask
        self.reg = torch.zeros_like(layer.weight.data)

        self.pr_mask = self.pr_mask.to(device)
        self.kp_mask = self.kp_mask.to(device)
        self.reg = self.reg.to(device)

    def _find_init_pruning_masks(self) -> torch.Tensor:
        weight_tensor = self.layer.weight.data
        pr_mask = 1 - MASK_FUNC_MAP[self.is_conv](weight_tensor, self.block_dimension, self.pr_ratio)
        return pr_mask

    def update_reg_single_layer(self, epsilon_lambda):
        self.reg += self.pr_mask * epsilon_lambda * torch.ones_like(self.reg)

    def apply_reg_single_layer(self):
        l2_grad = self.reg * self.layer.weight
        self.layer.weight.grad += l2_grad

    def get_kp_weight_norm(self) -> float:
        if self.pr_ratio == 0:
            return torch.norm(self.layer.weight.data) ** 2
        else:
            return torch.norm(self.kp_mask * self.layer.weight.data) ** 2

    def get_pr_weight_norm(self) -> float:
        if self.pr_ratio == 0:
            return 0
        else:
            return torch.norm(self.pr_mask * self.layer.weight.data) ** 2

    def finished_updating_reg_single_layer(self, reg_upper_limit) -> bool:
        return self.reg.max() > reg_upper_limit

    def get_sparsity_ratio(self):
        return self.pr_mask.sum().item() / self.pr_mask.numel()


class GRegIIArgs(GRegIArgs):
    def __init__(self, picking_ceiling, weight_decay, *args, **kwargs):
        super(GRegIIArgs, self).__init__(*args, **kwargs)
        self.weight_decay = weight_decay
        self.picking_ceiling = picking_ceiling

    def _find_init_pruning_masks(self) -> torch.Tensor:
        return torch.ones_like(self.layer.weight.data)

    def update_reg_single_layer(self, epsilon_lambda):
        if torch.sum(self.kp_mask) == 0 and (self.pr_mask * self.reg).max() > self.picking_ceiling:
            self.pr_mask = MASK_FUNC_MAP[self.is_conv](self.layer.weight.data, self.block_dimension, self.pr_ratio)
            self.kp_mask = 1 - self.pr_mask

        self.reg += self.pr_mask * epsilon_lambda * torch.ones_like(self.reg)
        self.reg += self.kp_mask * (-self.weight_decay) * torch.ones_like(self.reg)  # todo finish updating function


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
            epsilon_lambda,

            logger,

            # loss function
            criterion=nn.CrossEntropyLoss(),

            # bookkeeping
            log_directory='./greg1_pruning_logs',
            save_interval=1,

            block_size_mode='min',

            init_pr_over_kp_threshold=1,
            init_model_sparsity=0.5,
            layer_sparsity_delta=0.1
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

        self.weight_decay = weight_decay

        self.criterion = criterion
        self.reg_ceiling = reg_ceiling
        self.update_reg_interval = update_reg_interval
        self.epsilon_lambda = epsilon_lambda
        self.valid_block_dims = torch.load(valid_block_pruning_path)
        self.pr_ratio = pr_ratio
        self.block_size_mode = block_size_mode
        self.pr_over_kp_weight_norm_history = []
        self.log_directory = log_directory
        self.save_interval = save_interval

        self.init_pr_over_kp_threshold = init_pr_over_kp_threshold
        self.init_model_sparsity = init_model_sparsity
        self.layer_sparsity_delta = layer_sparsity_delta

        self.target_layers, self.target_model_sparsity = self._register_layers()

    def _get_sparsity_ratio(self, name, module, block_dimension):
        self.logger.info(f'getting sparsity ratio by flooding {name}')

        mask_func = MASK_FUNC_MAP[isinstance(module, nn.Conv2d)]
        pr_ratio = self.init_model_sparsity
        pr_mask = 1 - mask_func(module.weight.data, block_dimension, pr_ratio)
        kp_mask = 1 - pr_mask
        pr_weight_norm = torch.norm(pr_mask * module.weight.data) ** 2
        kp_weight_norm = torch.norm(kp_mask * module.weight.data) ** 2
        pr_over_kp = pr_weight_norm / kp_weight_norm

        if pr_over_kp < self.init_pr_over_kp_threshold:
            """
            use the current sparsity level if the resultant pr_over_kp satisfies the threshold requirement 
            """
            return pr_ratio

        """
        Otherwise reduce the pr_ratio by sparsity_delta until pkr reaches given threshold
        """
        while pr_over_kp > self.init_pr_over_kp_threshold:
            pr_ratio -= self.layer_sparsity_delta
            pr_mask = 1 - mask_func(module.weight.data, block_dimension, pr_ratio)
            kp_mask = 1 - pr_mask
            pr_weight_norm = torch.norm(pr_mask * module.weight.data) ** 2
            kp_weight_norm = torch.norm(kp_mask * module.weight.data) ** 2
            pr_over_kp = pr_weight_norm / kp_weight_norm

        return pr_ratio

    def _register_layers(self) -> (dict, float):
        self.logger.info(f"Looking for layers to prune")
        target_layers = {}
        expected_model_sparsity = 0

        count = 0
        pr_over_kp_weight_norm = {}

        targets = {name: module for name, module in self.model.named_modules() if is_compute_layer(module)}
        for layer_idx, (name, module) in enumerate(targets.items()):
            block_dimension = self._get_block_dimension(module, name)
            pr_ratio = self._get_sparsity_ratio(name, module, block_dimension)

            target_layers[name] = GRegIArgs(
                layer=module,
                block_dimension=block_dimension,
                pr_ratio=pr_ratio,
                device=self.device
            )

            kp_weight = target_layers[name].get_kp_weight_norm()
            pr_weight = target_layers[name].get_pr_weight_norm()
            pr_over_kp = pr_weight / kp_weight

            actual_layer_sparsity = target_layers[name].get_sparsity_ratio()
            self.logger.info(f"Register layer name: {name}, "
                             f"shape: {module.weight.data.shape} "
                             f"using pr ratio {actual_layer_sparsity} "
                             f"block dims: {block_dimension} "
                             f"PrWeightNorm/KpWeightNorm={pr_over_kp}")
            pr_over_kp_weight_norm[name] = pr_over_kp
            count += 1
            weight_proportion = module.weight.data.numel() / self.total_param
            expected_model_sparsity += weight_proportion * actual_layer_sparsity

        self.pr_over_kp_weight_norm_history.append(pr_over_kp_weight_norm)
        self.logger.info(f"Expected model sparsity: {expected_model_sparsity}")
        return target_layers, expected_model_sparsity

    def _get_block_dimension(self, module, name):
        block_dims_candidate = self.valid_block_dims[name]
        block_sizes = np.array([v[0] * v[1] for v in block_dims_candidate if v != (1, 1)])
        if self.block_size_mode == 'min':
            self.logger.info("Select a non-unstructured candidate with the smallest block size")
            block_dimension = block_dims_candidate[np.argmin(block_sizes)] if len(block_sizes) >= 1 else (1, 1)
        elif self.block_size_mode == 'max':
            self.logger.info("Select a non-unstructured candidate with the largest block size")
            block_dimension = block_dims_candidate[np.argmax(block_sizes)] if len(block_sizes) >= 1 else (1, 1)
        elif self.block_size_mode == 'unstructured':
            self.logger.info("Set block dimension to be 1x1")
            block_dimension = (1, 1)
        elif self.block_size_mode == 'regular':
            if isinstance(module, nn.Linear):
                num_row, num_col = module.weight.data.shape
                block_dimension = (num_row, 1)
            else:
                cout, cin, hk, wk = module.weight.data.shape
                if cout > cin:
                    block_dimension = (cout, hk * wk)
                else:
                    block_dimension = (1, cin * hk * wk)
        else:
            raise NotImplementedError
        return block_dimension

    def prune(self) -> nn.Module:
        self.model = self.model.train()
        num_iter = 0
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0

        while not self.finish_updating_reg_all_layer():
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.model.train()

                self.logger.info(f"Propagating Inputs @ Iter = {num_iter}")
                y_ = self.model(inputs)

                if num_iter % self.update_reg_interval == 0:
                    self.logger.info(f"Update Reg Term @ Iter = {num_iter}")
                    self.update_reg()

                self.logger.info(f"Backprop @ Iter = {num_iter}")
                loss = self.criterion(y_, targets)
                self.optimizer.zero_grad()
                loss.backward()

                self.logger.info(f"Apply Reg and Perform Gradient Descent @ Iter = {num_iter}")
                self.apply_reg()
                self.optimizer.step()

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
                    train_acc, train_loss = accuracy_evaluator.eval(
                        self.model, self.device, self.trainloader, self.criterion, print_acc=False
                    )

                    self.logger.info("TRAIN Acc1 = %.4f, Iter = %d @ Batch [%d]/[%d]" % (
                        train_acc, num_iter, batch_idx + 1, len(self.trainloader)))

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


class GRegPrunerII(GRegPrunerI):
    def __init__(self, picking_ceiling, *args, **kwargs):
        super(GRegPrunerII, self).__init__(*args, *kwargs)
        self.picking_ceiling = picking_ceiling

    def _register_layers(self) -> (dict, float):
        self.logger.info(f"Looking for layers to prune")
        target_layers = {}
        expected_model_sparsity = 0

        count = 0
        pr_over_kp_weight_norm = {}

        targets = {name: module for name, module in self.model.named_modules() if is_compute_layer(module)}
        for layer_idx, (name, module) in enumerate(targets.items()):
            block_dims_candidate = self.valid_block_dims[name]
            block_sizes = np.array([v[0] * v[1] for v in block_dims_candidate if v != (1, 1)])

            if self.block_size_mode == 'min':
                self.logger.info("Select a non-unstructured candidate with the smallest block size")
                block_dimension = block_dims_candidate[np.argmin(block_sizes)] if len(block_sizes) >= 1 else (1, 1)
            elif self.block_size_mode == 'max':
                self.logger.info("Select a non-unstructured candidate with the largest block size")
                block_dimension = block_dims_candidate[np.argmax(block_sizes)] if len(block_sizes) >= 1 else (1, 1)
            elif self.block_size_mode == 'unstructured':
                self.logger.info("Set block dimension to be 1x1")
                block_dimension = (1, 1)
            else:
                raise NotImplementedError

            pr_ratio = self._get_sparsity_ratio(name, module, layer_idx, block_dimension)

            target_layers[name] = GRegIIArgs(
                layer=module,
                block_dimension=block_dimension,
                pr_ratio=pr_ratio,
                device=self.device,
                picking_ceiling=self.picking_ceiling,
                weight_decay=self.weight_decay
            )

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


MASK_FUNC_MAP = {
    0: mask_linear_layer,
    1: mask_conv_layer
}
