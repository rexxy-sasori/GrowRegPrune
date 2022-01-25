import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

grasp_paths = glob.glob('Experiments/greg3_logs_*')
mb_paths = [p for p in grasp_paths if 'max' in p]
unstructured_paths = [p for p in grasp_paths if 'unstructured' in p]


def get_model_sparsity(ckpt):
    state_dict = ckpt['state_dict']
    num_ones = 0
    num_ele = 0

    sparsity_assignment = []
    for name, tensor in state_dict.items():
        if tensor.dim() == 4 or tensor.dim() == 2:
            mask = (tensor != 0).float()
            sparsity_assignment.append(1 - mask.sum().item() / mask.numel())
            num_ones += mask.sum().item()
            num_ele += mask.numel()

    return 1 - num_ones / num_ele, sparsity_assignment


def parse_data(log_dir):
    log_path = os.path.join(log_dir, 'my_log_info.log')
    ckpt_path = os.path.join(log_dir, 'finetune_ckpt.pt')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_sparsity, sparsity_assignment = get_model_sparsity(ckpt)

    test_accs = []
    test_acc_str = 'TEST ACC '
    with open(log_path, 'r') as f:
        lines = f.readlines()
        test_acc_lines = [l for l in lines if test_acc_str in l]
        for l in test_acc_lines:
            end_idx = l.index(' at ')
            start_idx = l.index('ACC')
            test_acc = float(l[start_idx + 4:end_idx])
            test_accs.append(test_acc)

    return model_sparsity, sparsity_assignment, 100 * np.array(test_accs)

# convergence plot
unstructured_data = {}
mb_data = {}
for p in unstructured_paths:
    model_sparsity, sparsity_assignment, test_accs = parse_data(p)
    unstructured_data[p] = (model_sparsity, sparsity_assignment, test_accs)

for p in mb_paths:
    model_sparsity, sparsity_assignment, test_accs = parse_data(p)
    mb_data[p] = (model_sparsity, sparsity_assignment, test_accs)

plt.figure()
for p, (model_sparsity, sparsity_assignment, test_accs) in unstructured_data.items():
    plt.plot(test_accs, markersize=3, label=f"Unstructured, Model Sparsity: {100 * model_sparsity:.2f}%")

for p, (model_sparsity, sparsity_assignment, test_accs) in mb_data.items():
    plt.plot(test_accs, markersize=3, label=f"Block, Model Sparsity: {100 * model_sparsity:.2f}%")

plt.grid()
plt.legend()
plt.xlim([90, 200])
plt.ylim([90, 95])
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.show()

plt.figure()
for p, (model_sparsity, sparsity_assignment, test_accs) in unstructured_data.items():
    plt.plot(sparsity_assignment, marker='o', markersize=5,
             label=f"Unstructured, Model Sparsity: {100 * model_sparsity:.2f}%")

for p, (model_sparsity, sparsity_assignment, test_accs) in mb_data.items():
    plt.plot(sparsity_assignment, marker='*', markersize=5,
             label=f"Block, Model Sparsity: {100 * model_sparsity:.2f}%")

plt.grid()
plt.legend()
plt.xlabel('Layer Idx')
plt.ylabel('Layer Sparsity')
plt.show()

