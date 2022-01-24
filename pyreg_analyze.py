import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

layer_names = [
    'features.0',
    'features.3',
    'features.7',
    'features.10',
    'features.14',
    'features.17',
    'features.20',
    'features.23',
    'features.27',
    'features.30',
    'features.33',
    'features.36',
    'features.40',
    'features.43',
    'features.46',
    'features.49',
    'classifier'
]


def parse_data(p):
    pr_over_kp = []
    test_accs = []
    train_accs = []
    layer_sparsity = []

    pr_over_kp_str = 'Pr/Kp='
    test_acc_str = 'TEST Acc1 = '
    train_acc_str = 'TRAIN Acc1 = '
    register_str = 'Register layer name:'
    expected_str = 'Expected model sparsity'
    with open(p, 'r') as f:
        lines = f.readlines()

        layer_pr_over_kp = [l for l in lines if pr_over_kp_str in l]
        test_acc_lines = [l for l in lines if test_acc_str in l]
        train_acc_lines = [l for l in lines if train_acc_str in l]
        layer_sparsity_lines = [l for l in lines if register_str in l]
        expected_sparsity_lines = [l for l in lines if expected_str in l]
        for idx, l in tqdm(enumerate(layer_pr_over_kp)):
            end_idx = l.index(', Lambda_Max')
            start_idx = l.index('Pr/Kp')
            num = float(l[start_idx + 6:end_idx])
            pr_over_kp.append(num)

        for idx, l in tqdm(enumerate(test_acc_lines)):
            end_idx = l.index(', Iter')
            start_idx = l.index(test_acc_str)
            num = float(l[start_idx + 12:end_idx])
            test_accs.append(num)

        for idx, l in tqdm(enumerate(train_acc_lines)):
            end_idx = l.index(', Iter')
            start_idx = l.index(train_acc_str)
            num = float(l[start_idx + 13:end_idx])
            train_accs.append(num)

        for l in tqdm(layer_sparsity_lines):
            end_idx = l.index(' block dims')
            start_idx = l.index('using pr ratio ')
            num = float(l[start_idx + 15:end_idx])
            layer_sparsity.append(num)

        end_idx = expected_sparsity_lines[0].index('\n')
        start_idx = expected_sparsity_lines[0].index('sparsity: ')
        expected_model_sparsity = float(expected_sparsity_lines[0][start_idx + 10:end_idx])

    return pr_over_kp, train_accs, test_accs, layer_sparsity, expected_model_sparsity


def plot_single_log_data(input_dir, output_dir):
    ret = parse_data(os.path.join(input_dir, 'my_log_info.log'))
    pr_over_kp, train_accs, test_accs, layer_sparsity, expected_model_sparsity = ret

    filename = os.path.split(input_dir.rstrip('/'))[-1]

    plt.figure()
    for idx, layer_name in enumerate(layer_names):
        plot_data = pr_over_kp[idx::len(layer_names)]
        plt.plot(plot_data, label=layer_name)

    # plt.ylim([0, 2])
    # plt.xlim([0, 20000])
    plt.legend(ncol=3)
    plt.grid()
    plt.xlabel('Gradient Update Epoch')
    plt.ylabel('PKR')
    plt.title(filename)
    plt.savefig(f"{output_dir}/pkr_{filename}.jpg")

    plt.figure()
    for idx, layer_name in enumerate(layer_names):
        plot_data = pr_over_kp[idx::len(layer_names)]
        plt.plot(plot_data, label=layer_name)

    plt.ylim([0, 0.01])
    plt.xlim([50000, 100000])
    plt.grid()
    plt.savefig(f"{output_dir}/pkr_zoom_{filename}.jpg")

    plt.figure()
    time_epochs = 9999 * np.arange(len(train_accs))
    plt.plot(time_epochs, train_accs, label='Train Acc', marker='o')
    plt.plot(time_epochs, test_accs, label='Test Acc', marker='o')
    plt.grid()
    plt.legend()
    plt.ylim([0.85, 1.01])
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.title(os.path.split(input_dir.rstrip('/'))[-1])
    plt.savefig(f"{output_dir}/acc_{filename}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', required=True)
    args = parser.parse_args()

    plot_single_log_data(args.log_path, 'plots')
