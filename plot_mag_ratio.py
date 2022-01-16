from matplotlib import pyplot as plt
import numpy as np
import torch

sps = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
remain_percentage = 100 - 100 * np.array(sps)
ratios = torch.load(f'unstructured/0.4.pt')
layer_names = list(ratios.keys())
num_layers = len(ratios)


def parse_data(num_row, num_col,dir_path):
    ret = np.zeros((num_row, num_col))
    for sp_idx, sp in enumerate(sps):
        ratios = torch.load(f"{dir_path}/{sp}.pt")
        for layer_idx, (layer_name, ratio) in enumerate(ratios.items()):
            ret[sp_idx, layer_idx] = ratio.item()

    return ret


data_unstructured = parse_data(len(sps), num_layers, 'unstructured')
data_block = parse_data(len(sps), num_layers, 'block')
data_block_extra = parse_data(len(sps), num_layers, 'block_extra_factor')

plt.figure()
plt.plot(remain_percentage, data_unstructured.max(1), label='unstructured', marker='o')
plt.plot(remain_percentage, data_block.max(1), label='block',  marker='o')
plt.plot(remain_percentage, data_block_extra.max(1), label='block with addition 100 factor', marker='o')
plt.legend()
plt.grid()
plt.xlabel('Remained Weight Percentage (%)')
plt.ylabel('Max PrWeightNorm/KpWeightNorm')
plt.show()