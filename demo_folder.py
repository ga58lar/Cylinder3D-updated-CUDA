# -*- coding:utf-8 -*-
# author: Ptzu
# @file: demo_folder.py

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")


def build_dataset(dataset_config,
                  data_dir,
                  grid_size=[480, 360, 32],
                  demo_label_dir=None):
    print(f"Building dataset with:")
    print(f"- data_dir: {data_dir}")
    print(f"- demo_label_dir: {demo_label_dir}")
    print(f"- label_mapping: {dataset_config['label_mapping']}")

    if demo_label_dir == '':
        imageset = "demo"
    else:
        imageset = "val"
    label_mapping = dataset_config["label_mapping"]

    SemKITTI_demo = get_pc_model_class('SemKITTI_demo')

    demo_pt_dataset = SemKITTI_demo(data_dir, imageset=imageset,
                              return_ref=True, label_mapping=label_mapping, demo_label_path=demo_label_dir)

    demo_dataset = get_model_class(dataset_config['dataset_type'])(
        demo_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    demo_dataset_loader = torch.utils.data.DataLoader(dataset=demo_dataset,
                                                     batch_size=1,
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=False,
                                                     num_workers=4)

    return demo_dataset_loader

def main(args):   
    pytorch_device = torch.device('cuda:0')
    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    data_dir = args.demo_folder
    demo_label_dir = args.demo_label_folder
    save_dir = args.save_folder + "/"

    demo_batch_size = 1
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    model_load_path = train_hypers['model_load_path']

    # Add these debug prints
    print(f"Model path exists: {os.path.exists(model_load_path)}")
    print(f"Data dir exists: {os.path.exists(data_dir)}")
    print(f"Label dir exists: {os.path.exists(demo_label_dir)}")

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    print("\nModel Architecture:")
    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"Total parameters: {total_params}")
    print("\nModel Structure:")
    for name, module in my_model.named_children():
        print(f"{name}:")
        print(f"  {module.__class__.__name__}")
        params = sum(p.numel() for p in module.parameters())
        print(f"  Parameters: {params}")

    # After loading model
    if os.path.exists(model_load_path):
        print("Loading model from:", model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)
        total_params = sum(p.numel() for p in my_model.parameters())
        print(f"Total parameters: {total_params}")
        # my_model.eval()
        # torch.set_grad_enabled(False)  # Ensure no gradients
        # print(f"Model training mode: {my_model.training}")
        
        # Validate model architecture
        print("\nModel Configuration:")
        print(f"Number of classes: {num_class}")
        print(f"Input features: {model_config['num_input_features']}")
        print(f"Output shape: {model_config['output_shape']}")

        # Check if weights are correctly loaded
        for name, param in my_model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Sum of weights: {param.sum().item()}")
    else:
        print("WARNING: Model file not found:", model_load_path)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    demo_dataset_loader = build_dataset(dataset_config, data_dir, grid_size=grid_size, demo_label_dir=demo_label_dir)
    with open(dataset_config["label_mapping"], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
        print("\nLabel Mapping Validation:")
        print(f"Number of classes in mapping: {len(semkittiyaml['learning_map'])}")
        print(f"Learning map: {semkittiyaml['learning_map']}")
    inv_learning_map = semkittiyaml['learning_map_inv']

    my_model.eval()
    hist_list = []
    demo_loss_list = []
    with torch.no_grad():
        for i_iter_demo, (_, demo_vox_label, demo_grid, demo_pt_labs, demo_pt_fea) in enumerate(
                demo_dataset_loader):
            if i_iter_demo == 0:
                # Validate input data
                print("\nInput Validation:")
                print(f"Point features range: ({demo_pt_fea[0].min():.3f}, {demo_pt_fea[0].max():.3f})")
                print(f"Label value counts:\n{np.unique(demo_pt_labs[0], return_counts=True)}")
            
            # Process batch
            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              demo_pt_fea]
            demo_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in demo_grid]
            demo_label_tensor = demo_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(demo_pt_fea_ten, demo_grid_ten, demo_batch_size)
            
            # Validate predictions
            if i_iter_demo == 0:
                print("\nPrediction Validation:")
                softmax_preds = torch.nn.functional.softmax(predict_labels, dim=1)
                print(f"Softmax predictions range: ({softmax_preds.min():.3f}, {softmax_preds.max():.3f})")
                
                # Chain max operations for each dimension
                max_per_class = softmax_preds.max(dim=4)[0]  # Max over last dimension
                max_per_class = max_per_class.max(dim=3)[0]  # Max over height
                max_per_class = max_per_class.max(dim=2)[0]  # Max over width
                max_per_class = max_per_class.max(dim=0)[0]  # Max over batch
                print(f"Max confidence per class: {max_per_class}")
                
                # Print prediction statistics
                print(f"\nPrediction Statistics:")
                print(f"Raw prediction shape: {predict_labels.shape}")
                print(f"Raw prediction range: ({predict_labels.min():.3f}, {predict_labels.max():.3f})")

                
            # Use softmax before computing loss
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels, dim=1).detach(), 
                                demo_label_tensor, ignore=ignore_label) + \
                   loss_func(predict_labels.detach(), demo_label_tensor)
                   
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            # Accumulate histograms properly
            for count, i_demo_grid in enumerate(demo_grid):
                hist = fast_hist_crop(predict_labels[count, demo_grid[count][:, 0], 
                                                   demo_grid[count][:, 1],
                                                   demo_grid[count][:, 2]], 
                                    demo_pt_labs[count],
                                    unique_label)
                hist_list.append(hist)
                
                # Map predictions back to original labels
                inv_labels = np.vectorize(inv_learning_map.__getitem__)(
                    predict_labels[count, demo_grid[count][:, 0], 
                                 demo_grid[count][:, 1], 
                                 demo_grid[count][:, 2]])
                inv_labels = inv_labels.astype('uint32')
                outputPath = save_dir + str(i_iter_demo).zfill(6) + '.label'
                inv_labels.tofile(outputPath)
                print("save " + outputPath)
                
            demo_loss_list.append(loss.detach().cpu().numpy())

    # Calculate IoU properly
    if demo_label_dir != '':
        total_hist = sum(hist_list)  # Combine all histograms
        iou = per_class_iu(total_hist)
        print('\nValidation Results:')
        print('Per class IoU:')
        for class_name, class_iou in zip(unique_label_str, iou):
            if not np.isnan(class_iou):
                print(f'{class_name:>20s} : {class_iou * 100:>5.2f}%')
            else:
                print(f'{class_name:>20s} : {"N/A":>5s}')
        
        val_miou = np.nanmean(iou) * 100
        print(f'\nMean IoU: {val_miou:.3f}%')
        print(f'Mean Loss: {np.mean(demo_loss_list):.3f}')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('--demo-folder', type=str, default='', help='path to the folder containing demo lidar scans', required=True)
    parser.add_argument('--save-folder', type=str, default='', help='path to save your result', required=True)
    parser.add_argument('--demo-label-folder', type=str, default='', help='path to the folder containing demo labels')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
