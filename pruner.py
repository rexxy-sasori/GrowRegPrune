import copy
import os
from copy import deepcopy

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nnutils.training_pipeline import accuracy_evaluator

from finetune import PresetLRScheduler
from finetune import tune_model


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
            layer_sparsity_delta=0.1,

            sparsity_assignment='rf'
    ):
        self.logger = logger

        self.device = device
        self.model = model.to(device)

        self.total_param = sum([p.numel() for p in model.state_dict().values() if p.dim() == 4 or p.dim() == 2])

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
        self.pr_over_kp_weight_norm_history = {}
        self.log_directory = log_directory
        self.save_interval = save_interval

        self.init_pr_over_kp_threshold = init_pr_over_kp_threshold
        self.init_model_sparsity = init_model_sparsity
        self.layer_sparsity_delta = layer_sparsity_delta

        self.sparsity_assignment = sparsity_assignment
        self.target_layers, self.target_model_sparsity = self._register_layers()

    def _get_sparsity_ratio(self, targets, block_dimensions):
        if self.sparsity_assignment == 'uniform':
            return self._uniform(targets, block_dimensions)
        elif self.sparsity_assignment == 'rf':
            return self._rf(targets, block_dimensions)
        elif self.sparsity_assignment == 'lamp':
            return self._lamp(targets, block_dimensions)
        else:
            raise NotImplementedError

    def _lamp(self, targets, block_dimensions):
        lamp_scores = {}
        ret = {}

        self.logger.info(f"compute LAMP score for each weight group")
        for name, module in targets.items():
            if isinstance(module, nn.Linear):
                weight_tensor = module.weight.data
            else:
                weight_tensor = module.weight.data
                cout, cin, hk, wk = weight_tensor.shape
                weight_tensor = weight_tensor.reshape(cout, cin * hk * wk)

            block_dimension = block_dimensions[name]
            blocks, num_blocks_row, num_blocks_col = matrix2blocks(
                weight_tensor, block_dimension[0], block_dimension[1]
            )
            sql2norm = torch.norm(blocks, dim=(-2, -1)) ** 2
            sorted_sql2norm, sorted_sql2norm_indices = torch.sort(sql2norm)

            denom = torch.sum(sorted_sql2norm) - torch.cumsum(sorted_sql2norm, 0) + sorted_sql2norm

            lamp_score = sorted_sql2norm / denom
            lamp_scores[name] = lamp_score

        all_lamp_scores = torch.cat(list(lamp_scores.values()))
        sorted_global_lamp, sorted_global_lamp_indices = torch.sort(all_lamp_scores)
        threshold = sorted_global_lamp[int(len(sorted_global_lamp) * 0.89)]
        self.logger.info(f"Determine layerwise sparsty threshold")
        for name, lamp_score in lamp_scores.items():
            ret[name] = torch.sum(lamp_score < threshold).item() / lamp_score.numel()

        return ret

    def _uniform(self, targets, block_dimensions):
        return {name: self.pr_ratio for name, _ in targets.items()}

    def _rf(self, targets, block_dimensions):
        ret = {}
        for name, module in targets.items():
            block_dimension = block_dimensions[name]
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

            ret[name] = pr_ratio
        return ret

    def _register_layers(self) -> (dict, float):
        self.logger.info(f"Looking for layers to prune")
        target_layers = {}
        expected_model_sparsity = 0

        count = 0

        targets = {name: module for name, module in self.model.named_modules() if is_compute_layer(module)}
        block_dimensions = {name: self._get_block_dimension(module, name) for name, module in targets.items()}
        sparsity_ratios = self._get_sparsity_ratio(targets, block_dimensions)
        for layer_idx, (name, module) in enumerate(targets.items()):
            target_layers[name] = GRegIArgs(
                layer=module,
                block_dimension=block_dimensions[name],
                pr_ratio=sparsity_ratios[name],
                device=self.device
            )

            kp_weight = target_layers[name].get_kp_weight_norm()
            pr_weight = target_layers[name].get_pr_weight_norm()
            pr_over_kp = pr_weight / kp_weight

            actual_layer_sparsity = target_layers[name].get_sparsity_ratio()
            self.logger.info(f"Register layer name: {name}, "
                             f"shape: {module.weight.data.shape} "
                             f"using pr ratio {actual_layer_sparsity} "
                             f"block dims: {block_dimensions[name]} "
                             f"PrWeightNorm/KpWeightNorm={pr_over_kp}")
            self.pr_over_kp_weight_norm_history[name] = [pr_over_kp]
            count += 1
            weight_proportion = module.weight.data.numel() / self.total_param
            expected_model_sparsity += weight_proportion * actual_layer_sparsity

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

        update_count = 0
        while update_count <= self.reg_ceiling / self.epsilon_lambda:
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.model.train()

                self.logger.info(f"Propagating Inputs @ Iter = {num_iter}")
                y_ = self.model(inputs)

                if num_iter % self.update_reg_interval == 0:
                    self.logger.info(f"Update Reg Term @ Iter = {num_iter}")
                    self.update_reg()
                    update_count += 1

                self.logger.info(f"Backprop @ Iter = {num_iter}")
                loss = self.criterion(y_, targets)
                self.optimizer.zero_grad()
                loss.backward()

                self.logger.info(f"Apply Reg and Perform Gradient Descent @ Iter = {num_iter}")
                self.apply_reg()
                self.optimizer.step()

                self.logger.info(f"Evaluating PrWeightNorm/KpWeightNorm @ Iter = {num_iter}")
                for name, layer_args in self.target_layers.items():
                    kp_weight = layer_args.get_kp_weight_norm()
                    pr_weight = layer_args.get_pr_weight_norm()
                    pr_over_kp = pr_weight / kp_weight
                    self.pr_over_kp_weight_norm_history[name].append(pr_over_kp)
                    self.logger.info(f"Name: {name}, Pr/Kp={pr_over_kp}, Lambda_Max={layer_args.reg.max()}")

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
                        {
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
                f'{name}: {self.pr_over_kp_weight_norm_history[name][0]} (before)'
                f'vs. {self.pr_over_kp_weight_norm_history[name][-1]} (after)'
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


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def grasp_score(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200,
                reinit=True):
    eps = 1e-10
    keep_ratio = 1 - ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []

    # rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal(layer.weight)
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N // 2])
        targets_one.append(dtarget[:N // 2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net.forward(inputs[:N // 2]) / T
        if print_once:
            # import pdb; pdb.set_trace()
            x = F.softmax(outputs)
            print(x)
            print(x.max(), x.min())
            print_once = False
        loss = F.cross_entropy(outputs, targets[:N // 2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:]) / T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs) / T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, (name, layer) in enumerate(net.named_modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[name] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return grads


class GraspPruner:
    def __init__(
            self,
            model,
            global_ratio,
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
            device,
            # optimizer
            lr_prune,
            momentum,
            weight_decay,
            # pruning block dimension
            valid_block_pruning_path,
            block_size_mode,
            logger,
            log_directory,
            # loss function
            criterion=nn.CrossEntropyLoss(),
    ):
        self.device = device
        self.model = model.to(device)

        self.total_param = sum([p.numel() for p in model.state_dict().values() if p.dim() == 4 or p.dim() == 2])

        self.global_ratio = global_ratio
        self.trainloader = trainloader
        self.testloader = testloader

        self.lr_prune = lr_prune
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr_prune,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.valid_block_dims = torch.load(valid_block_pruning_path)
        self.logger = logger
        self.criterion = criterion

        self.grasp_score = grasp_score(
            self.model, self.global_ratio, self.trainloader, self.device, 10, 10, 1, 200, True
        )

        self.block_size_mode = block_size_mode
        self.log_directory = log_directory

        self._register_mask()

    def _register_mask(self):
        norm_factor = sum([torch.sum(v).item() for v in self.grasp_score.values()])

        all_scores = []
        block_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                score = self.grasp_score[name]
                blockdim = self._get_block_dimension(module, name)
                if score.dim() == 4:
                    cout, cin, hk, wk = score.shape
                    img2col_score = score.reshape(cout, cin * hk * wk)
                else:
                    img2col_score = score

                block_score_sep, num_blocks_row, num_blocks_col = matrix2blocks(
                    img2col_score, blockdim[0], blockdim[1]
                )

                block_score_sep /= norm_factor
                block_score = torch.sum(block_score_sep, dim=(-2, -1))
                block_scores[name] = (block_score_sep, num_blocks_row, num_blocks_col, blockdim)
                all_scores.append(block_score)

        all_scores = torch.cat(all_scores)
        num_params_to_rm = int(len(all_scores) * self.global_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)

        total_sparsity = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                block_score_sep, num_blocks_row, num_blocks_col, blockdim = block_scores[name]
                block_mask = (torch.sum(block_score_sep, dim=(-2, -1)) <= threshold[-1]).float()
                block_mask = block_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(block_score_sep)
                mask = blocks2matrix(block_mask, num_blocks_row, num_blocks_col, blockdim[0], blockdim[1])

                if isinstance(module, nn.Conv2d):
                    cout, cin, hk, wk = module.weight.data.shape
                    mask = mask.reshape(cout, cin, hk, wk)

                layer_sparsity = 1 - mask.sum().item() / mask.numel()
                total_sparsity += module.weight.data.numel() / self.total_param * layer_sparsity
                self.logger.info(f"Sparsity Ratio for {name}: {layer_sparsity}")
                self.logger.info(f"mask shape: {mask.shape}, weight shape: {module.weight.data.shape}")
                setattr(module, 'mask', mask)

        self.logger.info(f"Expected Model Sparsity: {total_sparsity}")

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

    def prune(self):
        num_epochs = 200
        lr_schedule = {
            0: self.lr_prune,
            int(num_epochs * 0.5): self.lr_prune * 0.1,
            int(num_epochs * 0.75): self.lr_prune * 0.01
        }
        lr_scheduler = PresetLRScheduler(lr_schedule)

        tune_model(
            criterion=self.criterion,
            device=self.device,
            logger=self.logger,
            lr_scheduler=lr_scheduler,
            num_epochs=num_epochs,
            optimizer=self.optimizer,
            output_dir=self.log_directory,
            sparse_model=self.model,
            testloader=self.testloader,
            trainloader=self.trainloader,
            update_freq=10
        )


MASK_FUNC_MAP = {
    0: mask_linear_layer,
    1: mask_conv_layer
}
