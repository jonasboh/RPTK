# Implementation DenseNet
# - cropping - train on cropped img
# - segmentation 2nd channel - add mask to training
# - aug - data augumentations on/off
# - batch - batch normalization on/off
# - image transformation - running on


# +
# import logging
import os
import sys
import numpy as np
import random
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist  # for multi CPU training
#from ignite.engine import Engine, Events
#from ignite.handlers import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resize, Resized, \
    ScaleIntensityd, RandFlipd, RandZoomd, Spacing, Rand3DElasticd, Lambdad
from monai.data.meta_tensor import MetaTensor

# -
# from medcam import medcam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage import zoom

# +
import SimpleITK as sitk


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def write_nifti_file(array, out_file_name, path):
    """
    Write a NIfTI file with the header information from an existing file.
    :param array: numpy.ndarray The array is to be saved to a NIfTI file.
    :param header_file_path : str The path to the existing NIfTI file that contains the header information.
    :param output_file_path : str The path where the output NIfTI file should be saved.
    """

    # Load the existing header information from the header file

    sitk_img = sitk.ReadImage(path)

    if type(array) == list:
        for arr in array:
            sitk_img_out = sitk.GetImageFromArray(arr.astype(np.uint16))
            sitk_img_out.CopyInformation(sitk_img)
            print("ERROR: Need to implement multiple filenames for multiple images")
    else:
        sitk_img_out = sitk.GetImageFromArray(array.astype(np.uint16))
        sitk_img_out.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_img_out, out_file_name, useCompression=True)


def z_score_normalize(image_path):
    """
    Write normalized Images via z-score normalization
    :param image_path:
    """

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    mean_value = image_array.mean()
    std = image_array.std()
    transformed_image_array = (image_array - mean_value) / (max(std, 1e-8))
    transformed_image_array = transformed_image_array.astype(np.uint16)
    transformed_image_array = np.nan_to_num(transformed_image_array)

    # generate file name
    transformed_file_name = os.path.basename(image_path)[:-len(".nii.gz")] + "_z_score_normalized.nii.gz"

    # write image to preprocessed images  array, out_file_name, path
    write_nifti_file(transformed_image_array, self.outpath + transformed_file_name, image_path)


class DenseNetBC(nn.Module):
    def __init__(self, growth_rate=40, block_config=(6, 12, 48, 32), bn_size=4, drop_rate=0, num_classes=2):
        super(DenseNetBC, self).__init__()

        num_init_features = 2 * growth_rate

        print("Generating features ...")
        print("num_init_features", num_init_features)

        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1,
                            out_channels=num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False),
            torch.nn.BatchNorm3d(num_features=num_init_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        num_layers = 1

        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            if i != len(block_config) - 1:
                trans = self._make_transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', torch.nn.BatchNorm3d(num_features))
        self.classifier = torch.nn.Linear(num_features, num_classes)

        num_features += num_layers * growth_rate
        print("num_features", num_features)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm3d) or isinstance(m, torch.nn.GroupNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def _make_dense_block(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        layers = []
        for i in range(num_layers):
            layer = self._make_dense_layer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        return torch.nn.Sequential(*layers)

    def _make_dense_layer(self, num_input_features, growth_rate, bn_size, drop_rate):
        layers = []
        layers.append(torch.nn.BatchNorm3d(num_input_features))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        layers.append(torch.nn.BatchNorm3d(bn_size * growth_rate))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(
            torch.nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            layers.append(torch.nn.Dropout(drop_rate))
        return torch.nn.Sequential(*layers)

    def _make_transition(self, num_input_features, num_output_features):
        return torch.nn.Sequential(
            torch.nn.BatchNorm3d(num_input_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            torch.nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        features = self.features(x)
        # Debug: Print feature shape after each dense block and transition layer
        for name, layer in self.features.named_children():
            features = layer(features)
            # print(f"Layer {name}: {features.shape}")
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


# Hook functions with a list to capture outputs
def get_first_layer_features(module, input, output, outputs_list):
    outputs_list.append(output)


def get_last_layer_features(module, input, output, outputs_list):
    outputs_list.append(output)


def perform_forward_pass(model, scan):
    # Containers for captured outputs
    first_layer_output = []
    last_layer_output = []

    # Register hooks
    first_layer = model.features[0]
    last_layer = model.class_layers

    first_layer.register_forward_hook(
        lambda module, input, output: get_first_layer_features(module, input, output, first_layer_output))
    last_layer.register_forward_hook(
        lambda module, input, output: get_last_layer_features(module, input, output, last_layer_output))

    # Perform the forward pass
    with torch.no_grad():
        _ = model(scan)

    # Extract the captured outputs
    first_features = first_layer_output[0] if first_layer_output else None
    last_features = last_layer_output[0] if last_layer_output else None

    return first_features, last_features


def save_attention_map(attention_map, foldername):
    os.makedirs(foldername, exist_ok=True)
    num_slices = attention_map.size(1)
    for i in range(num_slices):
        plt.imshow(attention_map[0, i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(foldername, f'slice_{i}.png'))
        plt.close()


#### PREPROCESSING ####

def load_nifti_image(file_path):
    return nib.load(file_path)


def save_nifti_image(image, affine, save_path):
    nib.save(nib.Nifti1Image(image, affine), save_path)


def crop_to_mask(image, mask):
    coords = np.array(np.nonzero(mask))
    x_min, y_min, z_min = np.min(coords, axis=1)
    x_max, y_max, z_max = np.max(coords, axis=1)
    cropped_image = image[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    return cropped_image


def z_score_normalization(image):
    scaler = StandardScaler()
    flat_image = image.flatten().reshape(-1, 1)
    normalized_image = scaler.fit_transform(flat_image).reshape(image.shape)
    return normalized_image


def resize_image(image, target_size, cropping):
    if cropping:
        # zoom into the cropped region to resize the image
        current_size = image.shape
        scale_factors = [t / c for t, c in zip(target_size, current_size)]
        resized_image = zoom(image, scale_factors, order=1)
        image = resized_image
    return image


def process_single_image(index, row, output_folder, spacing_dim=(1, 1, 1), resize_dim=(96, 96, 96), cropping=True,
                         normalize=True):
    """
    Doing normalize, corpping, spacing, resample, resize
    """

    image_path = row['Image']
    mask_path = row['Mask']
    unique_id = str(row['ID'])  # Assuming there is an "ID" column in the DataFrame

    # Load image and mask
    image_nii = load_nifti_image(image_path)
    mask_nii = load_nifti_image(mask_path)

    # Get data and ensure they are NumPy arrays
    image = np.asanyarray(image_nii.dataobj)
    mask = np.asanyarray(mask_nii.dataobj)
    affine = image_nii.affine

    if normalize:
        # Z-score normalization
        normalized_image = z_score_normalization(image)
        image = normalized_image

    if cropping:
        # Crop around the mask
        cropped_image = crop_to_mask(image, mask)
        image = cropped_image

        # Convert to MetaTensor with affine matrix
        cropped_meta_tensor = MetaTensor(cropped_image, affine)
        img_meta_tensor = cropped_meta_tensor
    else:
        img_meta_tensor = MetaTensor(image, affine)

    # Resample to spacing_dim (e.g. 1x1x1 mm) spacing
    spacing_transform = Spacing(pixdim=spacing_dim, mode='bilinear')

    try:
        resampled_image = spacing_transform(img_meta_tensor)
    except np.linalg.LinAlgError:
        print(f"LinAlgError encountered for image ID {unique_id}. Using identity matrix.")

        # Use identity affine if the original affine causes issues
        identity_affine = np.eye(4)
        cropped_meta_tensor = MetaTensor(image, identity_affine)
        img_meta_tensor = cropped_meta_tensor
        resampled_image = spacing_transform(img_meta_tensor)

    resampled_image_np = resampled_image.numpy()
    resampled_affine = resampled_image.affine

    resized_image = resize_image(resampled_image_np, resize_dim, cropping)

    affine = resampled_image.affine

    original_filename = os.path.basename(image_path)
    filename_wo_ext = original_filename[:-len(".nii.gz")]
    processed_image_filename = f"{unique_id}_{filename_wo_ext}_preprocessed.nii.gz"
    processed_image_path = os.path.join(output_folder, processed_image_filename)

    file_counter = 1
    while os.path.exists(processed_image_path):
        processed_image_filename = f"{unique_id}_{filename_wo_ext}_preprocessed_{file_counter}.nii.gz"
        processed_image_path = os.path.join(output_folder, processed_image_filename)
        file_counter += 1

    save_nifti_image(resized_image, affine, processed_image_path)

    return index, processed_image_path


def preprocess_images(df, output_folder, num_workers=4, spacing_dim=(1, 1, 1), resize_dim=(96, 96, 96), cropping=True,
                      normalize=True):
    preprocessed_csv_path = os.path.join(output_folder, 'preprocessed_data.csv')

    if os.path.exists(preprocessed_csv_path):
        preprocessed_df = pd.read_csv(preprocessed_csv_path)
    else:
        # Create a new column for processed paths
        df['Processed_Image'] = ""
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, index, row, output_folder, spacing_dim, resize_dim, cropping, normalize): index for index, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df), desc="Preprocessing"):
                index, processed_path = future.result()
                df.loc[index, 'Processed_Image'] = processed_path
        
        preprocessed_df = df.copy()
        preprocessed_df['Image'] = preprocessed_df['Processed_Image']
        preprocessed_df.drop(columns=['Processed_Image'], inplace=True)
        preprocessed_df.to_csv(preprocessed_csv_path, index=False)
        
    return preprocessed_df

# -

def main(input_csv: str,
         output_dir: str,
         run_name: str,
         project_name: str,
         split_json: str, # New argument for the JSON split file
         device: str = "",
         num_classes: int = 2,
         loss="CrossE",
         batch_size=7,
         min_lr=1e-6,
         val_interval=1,
         max_epoch=200,
         init_lr=1e-4,
         n_cpu=1,
         z_norm=False,
         early_stopping=False,
         patience=3,
         min_delta=0.1,
         batch_norm=True,
         data_augumentation=True,
         attention_map_generation=False,
         add_mask2train=False,
         cropping=False,
         spacing=(1, 1, 1),
         wandb_key=None,
         resize_dim=(96, 96, 96)):
    """
    Performing DenseNet Classification
    """

    # ... (rest of the initial setup code remains the same)
    seed = 1234
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if not wandb_key is None:
        wandb.login(key=wandb_key)
    else:
        print("No detailed results tracking... If you want you can create a WanDB Account and provide your wandb_key. See <help> for more information.")

    if output_dir.endswith("/"):
        output_dir = output_dir[:-1]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    monai.config.print_config()

    if batch_norm:
        args = {"norm": "batch"}
        resnet_args = {"norm_type": ("batch", {"affine": True})}
    else:
        args = {"norm": ""}
        resnet_args = {"norm_type": ""}

    if add_mask2train:
        print("Use Channels: 2")
        # Models to train:
        ResNet18 = monai.networks.nets.ResNet(block="basic", layers=[2, 2, 2, 2], spatial_dims=3, 
                                              n_input_channels=2, block_inplanes=[64, 128, 256, 512], num_classes=num_classes)
        ResNet200 = monai.networks.nets.ResNet(block="basic", layers=[3, 24, 36, 3], spatial_dims=3, 
                                              n_input_channels=2, block_inplanes=[64, 128, 256, 512], num_classes=num_classes)
        DenseNet121 = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=num_classes, **args)
        DenseNet169 = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=2, out_channels=num_classes, **args)
        DenseNet201 = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=2, out_channels=num_classes, **args)
        DenseNet264 = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=2, out_channels=num_classes, **args)
    else:
        print("Use Channels: 1")
        # Models to train:
        ResNet18 = monai.networks.nets.ResNet(block="basic", layers=[2, 2, 2, 2], spatial_dims=3, 
                                              n_input_channels=1,  block_inplanes=[64, 128, 256, 512], num_classes=num_classes)
        ResNet200 = monai.networks.nets.ResNet(block="basic", layers=[3, 24, 36, 3], spatial_dims=3, 
                                              n_input_channels=1, block_inplanes=[64, 128, 256, 512], num_classes=num_classes)
        HighResNet = monai.networks.nets.HighResNet(spatial_dims=3, in_channels=1, out_channels=num_classes, **resnet_args)
        DenseNet121 = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes, **args)
        DenseNet169 = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1, out_channels=num_classes, **args)
        DenseNet201 = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1, out_channels=num_classes, **args)
        DenseNet264 = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=num_classes, **args)

    models = {
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "DenseNet201": DenseNet201,
        "DenseNet264": DenseNet264,
        "ResNet18":ResNet18,
        "ResNet200":ResNet200,
    }

    df = pd.read_csv(input_csv)
    
    if "Mask_Trasformation" in df.columns:
            df = df.loc[df["Mask_Trasformation"].isnull()]
    
    if add_mask2train:
        df = df.drop_duplicates(subset=["Image", "Mask"])
    else:
        df = df.drop_duplicates(subset="Image")
    
    if cropping:
        resize_dim = (32, 32, 32)

    preprocessed_folder = os.path.join(output_dir, "preprocessed")
    Path(preprocessed_folder).mkdir(parents=True, exist_ok=True)

    df = preprocess_images(df,
                           preprocessed_folder,
                           num_workers=n_cpu,
                           spacing_dim=spacing,
                           resize_dim=resize_dim,
                           cropping=cropping,
                           normalize=z_norm)
    
    df.set_index("ID", inplace=True)

    # Load splits from JSON file
    with open(split_json, 'r') as f:
        splits = json.load(f)
    
    test_ids = splits['test']
    X_test = df.loc[test_ids]
    y_test = X_test["Prediction_Label"]

    train_val_df = df.drop(test_ids)
    
    # Prepare test files
    if add_mask2train:
        test_files = [{"img": row["Image"], "mask": row["Mask"], "label": row["Prediction_Label"], "ID": idx} 
                      for idx, row in X_test.iterrows()]
    else:
        test_files = [{"img": row["Image"], "label": row["Prediction_Label"], "ID": idx} 
                      for idx, row in X_test.iterrows()]

    folds = sorted(splits['train'].keys())
    num_folds = len(folds)

    ### Select Model ###
    for model_name in models:
        print("#" * 5, "Training", model_name, "#" * 5)
        
        if not wandb_key is None:
            run = wandb.init(
                project=project_name,
                notes="binary_training_with_json_split",
                tags=["DL_baseline"],
                name=run_name + "_" + model_name
            )
        
        output = os.path.join(output_dir, model_name)
        Path(output).mkdir(parents=True, exist_ok=True)
        
        if device == "":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {device}")

        all_test_predictions = []

        ######## Train ########
        for fold_name in folds:
            fold = int(fold_name.split('_')[-1])
            print("-" * 5, f"Training Fold {fold}", "-" * 5)
            
            model_path = os.path.join(output, f"best_metric_model_classification3d_{fold}.pth")
            
            if not os.path.exists(model_path):
                train_ids, val_ids = splits['train'][fold_name]
                
                X_train = df.loc[train_ids]
                y_train = X_train["Prediction_Label"]
                X_val = df.loc[val_ids]
                y_val = X_val["Prediction_Label"]
                
                if add_mask2train:
                    train_files = [{"img": row["Image"], "mask": row["Mask"], "label": row["Prediction_Label"], "ID": idx} 
                                   for idx, row in X_train.iterrows()]
                    val_files = [{"img": row["Image"], "mask": row["Mask"], "label": row["Prediction_Label"], "ID": idx} 
                                 for idx, row in X_val.iterrows()]
                else:
                    train_files = [{"img": row["Image"], "label": row["Prediction_Label"], "ID": idx} 
                                   for idx, row in X_train.iterrows()]
                    val_files = [{"img": row["Image"], "label": row["Prediction_Label"], "ID": idx} 
                                 for idx, row in X_val.iterrows()]
                
                # Data Augmentation and Transforms
                # Ensure labels are always torch.long regardless of CSV dtype
                label_to_long = Lambdad(keys="label", func=lambda x: torch.as_tensor(x, dtype=torch.long))
                if data_augumentation:
                    train_transforms = Compose(
                        [
                            LoadImaged(keys=["img"], ensure_channel_first=True),
                            ScaleIntensityd(keys=["img"]),
                            Resized(keys=["img"], spatial_size=resize_dim),
                            RandRotate90d(keys=["img"], prob=0.2, spatial_axes=[0, 2]),
                            RandFlipd(keys=["img"], prob=0.2, spatial_axis=0),
                            RandFlipd(keys=["img"], prob=0.2, spatial_axis=1),
                            RandFlipd(keys=["img"], prob=0.2, spatial_axis=2),
                        ]
                    )
                else:
                    train_transforms = Compose(
                        [
                            LoadImaged(keys=["img"], ensure_channel_first=True),
                            ScaleIntensityd(keys=["img"]),
                            Resized(keys=["img"], spatial_size=resize_dim)
                        ]
                    )

                val_transforms = Compose(
                    [
                        LoadImaged(keys=["img"], ensure_channel_first=True),
                        ScaleIntensityd(keys=["img"]),
                        Resized(keys=["img"], spatial_size=resize_dim)
                    ]
                )
                
                # Dataloaders
                train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, pin_memory=torch.cuda.is_available(), drop_last=True)
                
                val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
                val_loader = DataLoader(val_ds, batch_size=batch_size*2, num_workers=n_cpu, pin_memory=torch.cuda.is_available())

                model = models[model_name].to(device).to(torch.float32)
                
                loss_function = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), init_lr)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=(max_epoch + 1), T_mult=1, eta_min=min_lr)
                
                auc_metric = ROCAUCMetric()
                best_metric = -1
                best_metric_epoch = -1
                
                if early_stopping:
                    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
                
                # Training loop
                for epoch in range(max_epoch):
                    model.train()
                    epoch_loss = 0
                    for batch_data in train_loader:
                        # Cast the input image to float32
                        inputs = batch_data["img"].to(device, dtype=torch.float32)
                        labels = batch_data["label"].to(device, dtype=torch.long)

                        optimizer.zero_grad()
                        
                        outputs = model(inputs)
                        loss = loss_function(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    epoch_loss /= len(train_loader)
                    scheduler.step()
                    
                    # Validation
                    model.eval()
                    val_loss = 0
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        for val_data in val_loader:
                            # Cast the validation image to float32
                            val_images = val_data["img"].to(device, dtype=torch.float32)
                            val_labels = val_data["label"].to(device, dtype=torch.long)

                            val_outputs = model(val_images)
                            val_loss += loss_function(val_outputs, val_labels).item()
                            y_pred = torch.cat([y_pred, val_outputs], dim=0)
                            y = torch.cat([y, val_labels], dim=0)
                    
                    val_loss /= len(val_loader)
                    
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    
                    post_pred = Compose([Activations(softmax=True)])
                    post_label = Compose([AsDiscrete(to_onehot=num_classes)])
                    y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                    y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                    
                    auc_metric(y_pred_act, y_onehot)
                    auc_result = auc_metric.aggregate()
                    auc_metric.reset()
                    
                    print(f"Epoch {epoch+1}/{max_epoch} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f} - Val AUC: {auc_result:.4f} - Val Acc: {acc_metric:.4f}")
                    
                    if not wandb_key is None:
                        wandb.log({f"train_loss_{fold}": epoch_loss, f"val_loss_{fold}": val_loss, f"val_auc_{fold}": auc_result, f"val_acc_{fold}": acc_metric, "epoch": epoch + 1})

                    if auc_result > best_metric:
                        best_metric = auc_result
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), model_path)
                        print("Saved new best model!")
                        
                    if early_stopping and early_stopper.early_stop(val_loss):
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Save validation predictions after training a fold
            print("Saving validation predictions...")
            model = models[model_name].to(device).to(torch.float32)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            val_ids = splits['train'][fold_name][1]
            X_val = df.loc[val_ids]
            if add_mask2train:
                 val_files = [{"img": row["Image"], "mask": row["Mask"], "label": row["Prediction_Label"], "ID": idx} 
                                 for idx, row in X_val.iterrows()]
            else:
                 val_files = [{"img": row["Image"], "label": row["Prediction_Label"], "ID": idx} 
                                 for idx, row in X_val.iterrows()]
            
            val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=n_cpu)
            
            val_predictions = []
            with torch.no_grad():
                for val_data in val_loader:
                    val_images = val_data["img"].to(device, dtype=torch.float32)
                    val_outputs = model(val_images)
                    probs = F.softmax(val_outputs, dim=1).cpu().numpy()[0]
                    val_predictions.append({
                        "ID": val_data["ID"][0],
                        "true_label": val_data["label"].item(),
                        "prob_class_0": probs[0],
                        "prob_class_1": probs[1]
                    })
            pd.DataFrame(val_predictions).to_csv(os.path.join(output, f"validation_predictions_fold_{fold}.csv"), index=False)
            
            ######## Test ########
            print("#" * 5, f"Test Model {model_name} from Fold {fold}", "#" * 5)
            test_transforms = Compose(
                [
                    LoadImaged(keys=["img"], ensure_channel_first=True),
                    ScaleIntensityd(keys=["img"]),
                    Resized(keys=["img"], spatial_size=resize_dim)
                ]
            )
            test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
            test_loader = DataLoader(test_ds, batch_size=1, num_workers=n_cpu)

            fold_test_predictions = []
            with torch.no_grad():
                for test_data in test_loader:
                    test_images = test_data["img"].to(device, dtype=torch.float32)
                    test_outputs = model(test_images)
                    probs = F.softmax(test_outputs, dim=1).cpu().numpy()[0]
                    fold_test_predictions.append({
                        "ID": test_data["ID"][0],
                        "true_label": test_data["label"].item(),
                        "prob_class_0": probs[0],
                        "prob_class_1": probs[1]
                    })

            fold_pred_df = pd.DataFrame(fold_test_predictions)
            fold_pred_df.to_csv(os.path.join(output, f"test_predictions_fold_{fold}.csv"), index=False)
            all_test_predictions.append(fold_pred_df.set_index('ID'))

        # Ensemble predictions
        if all_test_predictions:
            ensemble_df = pd.concat(all_test_predictions)
            prob_cols = ['prob_class_0', 'prob_class_1']
            # Corrected line: Only calculate the mean for probability columns
            mean_probs = ensemble_df.groupby('ID')[prob_cols].mean()
            
            # The rest of the code works as intended
            true_labels = ensemble_df.groupby('ID')['true_label'].first()
            ensemble_results = mean_probs.join(true_labels)
            ensemble_results.reset_index(inplace=True)
            # Reorder columns for clarity
            ensemble_results = ensemble_results[['ID', 'true_label', 'prob_class_0', 'prob_class_1']]
            ensemble_results.to_csv(os.path.join(output, "ensemble_test_predictions.csv"), index=False)
            print("Ensemble predictions saved.")
        
        if not wandb_key is None:
            wandb.finish()
            
    print("#" * 5, "All Experiments Done", "#" * 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply DenseNet on Images for classification.')
    parser.add_argument('--csv', metavar='CSV', type=str, required=True,
                        help='Path to csv file. Image: Path to Image and Prediction_Label: Label of the Image')
    parser.add_argument('--split_json', metavar='JSON', type=str, required=True,
                        help='Path to json file with train/test splits.')
    parser.add_argument('--device', metavar='Device', type=str, default="cuda",
                        help='Device using for training cuda or cpu')
    parser.add_argument('--project_name', metavar='PROJECT_NAME', type=str, default="DenseNet",
                        help='Name of the project for WanDB')
    parser.add_argument('--run_name', metavar='RUN_NAME', type=str, default="DenseNet",
                        help='Name of the run in the project for WanDB')
    parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str, default='.',
                        help='Output directory for plots and CSV file')
    parser.add_argument('--num_classes', metavar='NUM_CLASSES', type=int, default=2,
                        help='Number of classes to classify')
    parser.add_argument('--loss_function_name', metavar='LOSS', type=str, default="CrossE",
                        help='Loss function CrossEntropyLoss or BCELoss')
    parser.add_argument('--batch_size', metavar='BATCH_SIZE', type=int, default=7, help='Batch size for training')
    parser.add_argument('--min_lr', metavar='MIN_LR', type=float, default=1e-6, help='Minimum learning rate to adapt')
    parser.add_argument('--val_interval', metavar='Val_INT', type=int, default=1,
                        help='Interval of validation after each X epoch')
    parser.add_argument('--max_epoch', metavar='MAX_EPOCH', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--init_lr', metavar='INIT_LR', type=float, default=1e-4,
                        help='Initial learning rate to start with')
    parser.add_argument('--n_cpu', metavar='N_CPU', type=int, default=1, help='Number of CPUs for preprocessing')
    parser.add_argument('--proxy', metavar='Proxy', type=str, default=None, help='Use https proxy for wandb')
    parser.add_argument('--wandb_key', metavar='Key', type=str, default=None, help='Use wandb key to track your results')
    parser.add_argument('--z_norm', action='store_true', help='Enable Z-score normalization')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', metavar='PATIENCE', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--min_delta', metavar='MIN_DELTA', type=float, default=0.1, help='Early stopping min delta')
    parser.add_argument('--no_batch_norm', action='store_true', help='Disable batch normalization')
    parser.add_argument('--no_data_aug', action='store_true', help='Disable data augmentation')
    parser.add_argument('--attention_map_generation', action='store_true', help='Generate attention maps')
    parser.add_argument('--add_mask2train', action='store_true',
                        help='Add mask to training - csv should contain column Mask')
    parser.add_argument('--cropping', action='store_true',
                        help='Enable Cropping/Resample/Resize - csv should contain Mask and ID columns')
    parser.add_argument('--spacing', metavar='SPACE', type=lambda s: tuple(map(int, s.split(','))), default=(1, 1, 1),
                        help='Spacing for resampling e.g. 1,1,1')
    parser.add_argument('--resize_dim', metavar='RESIZE', type=lambda s: tuple(map(int, s.split(','))), default=(96, 96, 96),
                        help='Dimensions for resizing e.g. 96,96,96')

    
    
    
    args = parser.parse_args()
    
    if not args.proxy is None:
        os.environ["HTTPS_PROXY"] = args.proxy
    
    main(input_csv=args.csv,
         output_dir=args.output_dir,
         device=args.device,
         run_name=args.run_name,
         project_name=args.project_name,
         split_json=args.split_json,
         num_classes=args.num_classes,
         loss=args.loss_function_name,
         batch_size=args.batch_size,
         min_lr=args.min_lr,
         val_interval=args.val_interval,
         max_epoch=args.max_epoch,
         init_lr=args.init_lr,
         n_cpu=args.n_cpu,
         z_norm=args.z_norm,
         early_stopping=args.early_stopping,
         patience=args.patience,
         min_delta=args.min_delta,
         batch_norm=not args.no_batch_norm,
         data_augumentation=not args.no_data_aug,
         attention_map_generation=args.attention_map_generation,
         add_mask2train=args.add_mask2train,
         cropping=args.cropping,
         spacing=args.spacing,
         wandb_key=args.wandb_key,
         resize_dim=args.resize_dim)
