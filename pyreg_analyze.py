import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

p = 'greg1_pruning_logs/log_unstructured.txt'

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


pr_over_kp = []
test_accs = []
train_accs = []
timestamps = []

pr_over_kp_str = 'Pr/Kp='
test_acc_str = 'TEST Acc1 = '
train_acc_str = 'TRAIN Acc1 = '
forward_prop_str = 'Propagating Inputs'
with open(p, 'r') as f:
    lines = f.readlines()

    layer_pr_over_kp = [l for l in lines if pr_over_kp_str in l]
    test_acc_lines = [l for l in lines if test_acc_str in l]
    train_acc_lines = [l for l in lines if train_acc_str in l]
    forward_pro_lines = [l for l in lines if forward_prop_str in l]

    for idx, l in enumerate(layer_pr_over_kp):
        end_idx = l.index('\n')
        start_idx = l.index('Pr/Kp')
        num = float(l[start_idx+6:end_idx])
        pr_over_kp.append(num)

    for idx, l in enumerate(test_acc_lines):
        end_idx = l.index(', Iter')
        start_idx = l.index(test_acc_str)
        num = float(l[start_idx+12:end_idx])
        test_accs.append(num)

    for idx, l in enumerate(train_acc_lines):
        end_idx = l.index(', Iter')
        start_idx = l.index(train_acc_str)
        num = float(l[start_idx+13:end_idx])
        train_accs.append(num)

    for idx, l in enumerate(forward_pro_lines):
        end_idx = l.index(']')
        start_idx = l.index('[')
        time_stamp_str = l[start_idx+6:end_idx]
        timestamps.append(datetime.strptime(time_stamp_str, '%d %b %Y %H:%M:%S'))
pr_over_kp = np.array(pr_over_kp)

plt.figure()
for idx, layer_name in enumerate(layer_names):
    plot_data = pr_over_kp[idx::len(layer_names)]
    plt.plot(plot_data, label=layer_name)

plt.ylim([0, 5])
plt.legend(ncol=3)
plt.grid()
plt.xlabel('Gradient Update Epoch')
plt.ylabel('PKR @ 90%')
plt.title(p)
plt.show()

plt.figure()
for idx, layer_name in enumerate(layer_names):
    plot_data = pr_over_kp[idx::len(layer_names)]
    plt.plot(plot_data, label=layer_name)

plt.ylim([0, 0.05])
plt.xlim([10000, 20000])
plt.grid()
plt.show()

plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.grid()
plt.legend()
plt.ylim([0.88, 1.01])
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.title(p)
plt.show()


# time_increments = [(j-i).total_seconds() for j,i in zip(timestamps[1::], timestamps[0:-1])]
# plt.figure()
# plt.plot(time_increments[0::10])
# plt.grid()
# plt.ylabel('Seconds')
# plt.show()



