# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


def load_checkpoint(model_load_path, model):
    print("\nLoading checkpoint...")
    my_model_dict = model.state_dict()
    checkpoint = torch.load(model_load_path)

    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        pre_weight = checkpoint['model_state_dict']
    else:
        pre_weight = checkpoint

    print(f"Pretrained keys ({len(pre_weight.keys())}): {list(pre_weight.keys())[:5]}")
    print(f"Model keys ({len(my_model_dict.keys())}): {list(my_model_dict.keys())[:5]}")

    # Try to handle model parallel weights
    if any('module.' in k for k in pre_weight.keys()):
        print("Detected DataParallel weights, removing 'module.' prefix")
        pre_weight = {k.replace('module.', ''): v for k, v in pre_weight.items()}

    # Try different key matching strategies
    matched_keys = {}
    remaining_model_keys = set(my_model_dict.keys())

    # Strategy 1: Direct match
    for k_model in remaining_model_keys.copy():
        if k_model in pre_weight and pre_weight[k_model].shape == my_model_dict[k_model].shape:
            matched_keys[k_model] = pre_weight[k_model]
            remaining_model_keys.remove(k_model)

    # Strategy 2: Match without module prefix
    for k_model in remaining_model_keys.copy():
        k_model_clean = k_model.split('.')[-2:]  # Get last two parts of key
        k_model_clean = '.'.join(k_model_clean)
        for k_pre in pre_weight:
            if k_pre.endswith(k_model_clean) and pre_weight[k_pre].shape == my_model_dict[k_model].shape:
                matched_keys[k_model] = pre_weight[k_pre]
                remaining_model_keys.remove(k_model)
                break

    # Strategy 3: Match by shape
    shape_dict = {tuple(v.shape): k for k, v in pre_weight.items() if hasattr(v, 'shape')}
    for k_model in remaining_model_keys.copy():
        shape = tuple(my_model_dict[k_model].shape)
        if shape in shape_dict and shape_dict[shape] in pre_weight:
            matched_keys[k_model] = pre_weight[shape_dict[shape]]
            remaining_model_keys.remove(k_model)

    print(f"\nMatched {len(matched_keys)} layers")
    print(f"Unmatched {len(remaining_model_keys)} layers")
    print("\nFirst 5 matched layers:")
    for k in list(matched_keys.keys())[:5]:
        print(f"  {k}: {matched_keys[k].shape}")
    print("\nFirst 5 unmatched layers:")
    for k in list(remaining_model_keys)[:5]:
        print(f"  {k}: {my_model_dict[k].shape}")

    # Update model weights
    my_model_dict.update(matched_keys)
    model.load_state_dict(my_model_dict)

    return model


def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
