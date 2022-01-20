CUDA_VISIBLE_DEVICES=0 python main.py --block-size-mode unstructured --reg-mode 1 --sparsity-assignment uniform
CUDA_VISIBLE_DEVICES=1 python main.py --block-size-mode max --reg-mode 1 --sparsity-assignment uniform
CUDA_VISIBLE_DEVICES=2 python main.py --block-size-mode regular --reg-mode 1 --sparsity-assignment rf
CUDA_VISIBLE_DEVICES=3 python main.py --block-size-mode regular --reg-mode 1 --sparsity-assignment uniform