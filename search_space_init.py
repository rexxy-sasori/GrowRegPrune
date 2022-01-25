import itertools

import numpy as np
from torch import nn


def is_compute_layer(module):
    return isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)


DEFAULT_BLOCK_SEARCH_SPACE = {
    'br': np.arange(1, 5000),
    'bc': np.arange(1, 5000),
}


def is_block_size_candidate_valid(candidate: int, dim: int):
    if candidate > dim:
        return False
    if dim % candidate != 0:
        return False
    else:
        return True


def conv_unstructured_single(cout, cin, hk, wk, search_space):
    return [(1, 1)]


def conv_unstructured_channel(cout, cin, hk, wk, search_space):
    return [(1, hk * wk)]


def conv_filter(cout, cin, hk, wk, search_space):
    return [(1, cin * hk * wk)]


def conv_structured_channel(cout, cin, hk, wk, search_space):
    return [(cout, hk * wk)]


def conv_mixed_structured_channel_filter(cout, cin, hk, wk, search_space):
    flag = np.random.randint(0, 2)
    if flag == 0:
        return conv_structured_channel(cout, cin, hk, wk, search_space)
    else:
        return conv_filter(cout, cin, hk, wk, search_space)


def conv_mixed_unstructured_channel_filter(cout, cin, hk, wk, search_space):
    flag = np.random.randint(0, 2)
    if flag == 0:
        return conv_unstructured_channel(cout, cin, hk, wk, search_space)
    else:
        return conv_filter(cout, cin, hk, wk, search_space)


def conv_mix(cout, cin, hk, wk, search_space):
    return [
        (1, 1),  # unstructured
        (1, hk * wk),  # unstructured channel
        (cout, hk * wk),  # structured channel
        (1, cin * hk * wk),  # filter pruning
        (cout, 1),  # conv column pruning
    ]


def conv_block(cout, cin, hk, wk, search_space):
    possible_brs = np.array([b for b in search_space['br'] if is_block_size_candidate_valid(b, cout)])
    possible_bcs = np.array([b for b in search_space['bc'] if is_block_size_candidate_valid(b, cin * hk * wk)])

    possible_br_bc_set = list(itertools.product(possible_brs, possible_bcs))

    total_ele = cout * cin * hk * wk
    max_param_one_block = total_ele * 0.01
    # br_bc_candidates = possible_br_bc_set
    br_bc_candidates = list(filter(lambda x: x[0] * x[1] <= max_param_one_block, possible_br_bc_set))

    return br_bc_candidates


def fc_mix(num_row, num_column, search_space):
    return [
        (1, 1),
        (num_row, 1),
        (1, num_column)
    ]


def fc_unstructured_single(num_row, num_column, search_space):
    return [(1, 1)]


def fc_block(num_row, num_column, search_space):
    possible_brs = np.array([b for b in search_space['br'] if is_block_size_candidate_valid(b, num_row)])
    possible_bcs = np.array(np.array([b for b in search_space['bc'] if is_block_size_candidate_valid(b, num_column)]))
    possible_br_bc_set = list(itertools.product(possible_brs, possible_bcs))

    total_ele = num_row * num_column
    max_param_one_block = total_ele * 0.01
    # br_bc_candidates = possible_br_bc_set
    br_bc_candidates = list(filter(lambda x: x[0] * x[1] <= max_param_one_block, possible_br_bc_set))

    return br_bc_candidates


CONV_PRUNING_FUNC = {
    'unstructured': conv_unstructured_single,
    'unstructured_channel': conv_unstructured_channel,
    'filter_only': conv_filter,
    'structured_channel': conv_structured_channel,
    'mixed_structured_channel_filter': conv_mixed_structured_channel_filter,
    'mixed_unstructured_channel_filter': conv_mixed_unstructured_channel_filter,
    'block': conv_block,
    'existing': conv_mix
}

FC_PRUNING_FUNC = {
    'unstructured': fc_unstructured_single,
    'block': fc_block,
    'existing': fc_mix
}


def get_block_search_space_for_conv(weight: np.array, mode: str = 'default', search_space={}):
    cout, cin, hk, wk = weight.shape
    return CONV_PRUNING_FUNC[mode](cout, cin, hk, wk, search_space)


def get_block_search_space_fc(weight: np.array, mode: str = 'default', search_space={}):
    num_out_features, num_in_features = weight.shape
    return FC_PRUNING_FUNC[mode](num_out_features, num_in_features, search_space)


def get_block_search_space_single_layer(
        layer: nn.Module,
        conv_mode: str = 'unstructured',
        fc_mode: str = 'unstructured',
        search_space={}
):
    weight = layer.weight.data
    if isinstance(layer, nn.Conv2d):
        return get_block_search_space_for_conv(weight, conv_mode, search_space)
    elif isinstance(layer, nn.Linear):
        return get_block_search_space_fc(weight, fc_mode, search_space)


def get_search_space(usr_valid_brs=(), usr_valid_bcs=()):
    ret = {}
    if len(usr_valid_brs) == 0:
        ret['br'] = DEFAULT_BLOCK_SEARCH_SPACE['br']
    elif len(usr_valid_brs) != 0:
        ret['br'] = set(usr_valid_brs).intersection(set(DEFAULT_BLOCK_SEARCH_SPACE['br'].tolist()))

    if len(usr_valid_bcs) == 0:
        ret['bc'] = DEFAULT_BLOCK_SEARCH_SPACE['bc']
    elif len(usr_valid_brs) != 0:
        ret['bc'] = set(usr_valid_brs).intersection(set(DEFAULT_BLOCK_SEARCH_SPACE['bc'].tolist()))

    return ret


def get_block_search_space_model(
        model: nn.Module,
        conv_mode: str = 'unstructured',
        fc_mode: str = 'unstructured',
        usr_valid_brs=(),
        usr_valid_bcs=()
):
    ret = {}

    search_space = get_search_space(usr_valid_brs, usr_valid_bcs)

    for idx, (name, module) in enumerate(model.named_modules()):
        if idx == 0:
            continue

        if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            continue

        if is_compute_layer(module):
            ret[name] = get_block_search_space_single_layer(module, conv_mode, fc_mode, search_space)

    return ret
