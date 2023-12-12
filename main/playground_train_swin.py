import argparse
import json
import os
import random
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from distutils.util import strtobool
from tqdm import tqdm
from torchvision.models import resnet18, resnet50

from dfc_dataset import DFCDataset


def main():

    DATA = 'data/data_disini'

    # Data configurations:
    data_config = {
        'train_dir': DATA, # path to the training directory,  
        'val_dir': DATA, # path to the validation directory,
        'train_mode': 'validation', # can be one of the following: 'test', 'validation'
        'val_mode': 'test', # can be one of the following: 'test', 'validation'
        'num_classes': 8, # number of classes in the dataset.
        'clip_sample_values': True, # clip (limit) values
        'train_used_data_fraction': 1, # fraction of data to use, should be in the range [0, 1]
        'val_used_data_fraction': 1,
        'image_px_size': 224, # image size (224x224)
        'cover_all_parts_train': True, # if True, if image_px_size is not 224 during training, we use a random crop of the image
        'cover_all_parts_validation': True, # if True, if image_px_size is not 224 during validation, we use a non-overlapping sliding window to cover the entire image
        'seed': 42,
    }

    random.seed(data_config['seed'])
    np.random.seed(data_config['seed'])
    torch.manual_seed(data_config['seed'])
    torch.cuda.manual_seed_all(data_config['seed'])

    train_dataset = DFCDataset(
        data_config['train_dir'],
        mode=data_config['train_mode'],
        clip_sample_values=data_config['clip_sample_values'],
        used_data_fraction=data_config['train_used_data_fraction'],
        image_px_size=data_config['image_px_size'],
        cover_all_parts=data_config['cover_all_parts_train'],
        seed=data_config['seed'],
    )

    val_dataset = DFCDataset(
        data_config['val_dir'],
        mode=data_config['val_mode'],
        clip_sample_values=data_config['clip_sample_values'],
        used_data_fraction=data_config['val_used_data_fraction'],
        image_px_size=data_config['image_px_size'],
        cover_all_parts=data_config['cover_all_parts_validation'],
        seed=data_config['seed'],
    )

    DFC_map_clean = {
        0: "Forest",
        1: "Shrubland",
        2: "Grassland",
        3: "Wetlands",
        4: "Croplands",
        5: "Urban/Built-up",
        6: "Barren",
        7: "Water",
        255: "Invalid",
    }

    # Training configurations
    train_config = {
        's1_input_channels': 2,
        's2_input_channels': 13,
        'finetuning': True, # If false, backbone layers is frozen and only the head is trained
        'classifier_lr': 3e-6,
        'learning_rate': 0.00001,
        'adam_betas': (0.9, 0.999), 
        'weight_decay': 0.001,
        'dataloader_workers': 4,
        'batch_size': 16,
        'epochs': 5, 
        'target': 'dfc_label'
    }

    # path to the checkpoint
    checkpoint = torch.load(
        "checkpoints/swin_t.pth",
        map_location=torch.device('cpu')
    ) 
    weights = checkpoint["state_dict"]

    # Sentinel-1 stream weights
    s1_weights = {
        k[len("backbone1."):]: v for k, v in weights.items() if "backbone1" in k
    }

    # Sentinel-2 stream weights
    s2_weights = {
        k[len("backbone2."):]: v for k, v in weights.items() if "backbone2" in k
    }

    from Transformer_SSL.models.swin_transformer import DoubleSwinTransformerDownstream
    from utils import save_checkpoint_single_model, dotdictify
    from Transformer_SSL.models import build_model


    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu:0")

    device = torch.device('cpu')

    # Input channel size
    input_channels = train_config['s1_input_channels'] + train_config['s2_input_channels']

    # If you are using a uni-modal dataset, you can un-comment one of these lines, and comment the one above:
    # input_channels = train_config['s1_input_channels']
    # input_channels = train_config['s2_input_channels']

    with open("configs/backbone_config.json", "r") as fp:
        swin_conf = dotdictify(json.load(fp))

    s1_backbone = build_model(swin_conf.model_config)

    swin_conf.model_config.MODEL.SWIN.IN_CHANS = 13
    s2_backbone = build_model(swin_conf.model_config)

    s1_backbone.load_state_dict(s1_weights)
    s2_backbone.load_state_dict(s2_weights)

    class DoubleSwinTransformerClassifier(torch.nn.Module):
        def __init__(self, encoder1, encoder2, out_dim, device, freeze_layers=True):
            super(DoubleSwinTransformerClassifier, self).__init__()
            
            # If you're only using one of the two backbones, just comment the one you don't need
            self.backbone1 = encoder1
            self.backbone2 = encoder2

            self.device = device

            # add final linear layer
            self.fc = torch.nn.Linear(
                self.backbone2.num_features + self.backbone1.num_features,
                out_dim,
                bias=True,
            )

            # freeze all layers but the last fc
            if freeze_layers:
                for name, param in self.named_parameters():
                    if name not in ["fc.weight", "fc.bias"]:
                        param.requires_grad = False

        def forward(self, x):
            x1, _, _ = self.backbone1.forward_features(x["s1"].to(self.device))
            x2, _, _ = self.backbone2.forward_features(x["s2"].to(self.device))

            z = torch.cat([x1, x2], dim=1)
            z = self.fc(z)
            
            # If you're only using one of the two backbones, you may comment the lines above and use the following:
            # x1, _, _ = self.backbone1.forward_features(x["s1"].to(self.device))
            # z = self.fc(x1)

            return z

    model = DoubleSwinTransformerClassifier(
            s1_backbone, s2_backbone, out_dim=data_config['num_classes'], device=device
        )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="mean").to(device)

    if train_config['finetuning']:
        # train all parameters (backbone + classifier head)
        param_backbone = []
        param_head = []
        for p in model.parameters():
            if p.requires_grad:
                param_head.append(p)
            else:
                param_backbone.append(p)
            p.requires_grad = True
        # parameters = model.parameters()
        parameters = [
            {"params": param_backbone},  # train with default lr
            {
                "params": param_head,
                "lr": train_config['classifier_lr'],
            },  # train with classifier lr
        ]
        print("Finetuning")

    else:
        # train only final linear layer for SSL methods
        print("Frozen backbone")
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))


    optimizer = torch.optim.Adam(
        parameters,
        lr=train_config['learning_rate'],
        betas=train_config['adam_betas'],
        weight_decay=train_config['weight_decay'],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=train_config['dataloader_workers'],
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['dataloader_workers'],
    )

    from metrics import ClasswiseAccuracy
    from validation_utils import validate_all

    step = 0

    # Training loop

    for epoch in range(train_config['epochs']):
        # Model Training
        model.train()
        step += 1

        pbar = tqdm(train_loader)

        # track performance
        epoch_losses = torch.Tensor()
        metrics = ClasswiseAccuracy(data_config['num_classes'])

        for idx, sample in enumerate(pbar):

            if "x" in sample.keys():
                if torch.isnan(sample["x"]).any():
                    # some s1 scenes are known to have NaNs...
                    continue
            else:
                if torch.isnan(sample["s1"]).any() or torch.isnan(sample["s2"]).any():
                    # some s1 scenes are known to have NaNs...
                    continue
            
            # load input
            s1 = sample["s1"].to(device)
            s2 = sample["s2"].to(device)
            img = {"s1": s1, "s2": s2}
            
            # if you are using a unimodal dataset (s1 for example), you may comment the lines above and use the following:
            # s1 = sample["s1"].to(device)
            # img = {"s1": s1}
            
            # load target
            y = sample[train_config['target']].long().to(device)
            
            # model output
            y_hat = model(img)
            
            # loss computation
            loss = criterion(y_hat, y)
            
            # backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # get prediction 
            _, pred = torch.max(y_hat, dim=1)

            epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
            metrics.add_batch(y, pred)

            pbar.set_description(f"Epoch:{epoch}, Training Loss:{epoch_losses[-100:].mean():.4}")

        mean_loss = epoch_losses.mean()

        train_stats = {
                "train_loss": mean_loss.item(),
                "train_average_accuracy": metrics.get_average_accuracy(),
                "train_overall_accuracy": metrics.get_overall_accuracy(),
                **{
                    "train_accuracy_" + k: v
                    for k, v in metrics.get_classwise_accuracy().items()
                },
            }
        print(train_stats)

        if epoch % 2 == 0:  

            # Model Validation
            model.eval()
            pbar = tqdm(val_loader)

            # track performance
            epoch_losses = torch.Tensor()
            metrics = ClasswiseAccuracy(data_config['num_classes'])

            with torch.no_grad():
                for idx, sample in enumerate(pbar):
                    if "x" in sample.keys():
                        if torch.isnan(sample["x"]).any():
                            # some s1 scenes are known to have NaNs...
                            continue
                    else:
                        if torch.isnan(sample["s1"]).any() or torch.isnan(sample["s2"]).any():
                            # some s1 scenes are known to have NaNs...
                            continue
                    # load input
                    s1 = sample["s1"].to(device)
                    s2 = sample["s2"].to(device)
                    img = {"s1": s1, "s2": s2}

                    # if you are using a unimodal dataset (s1 for example), you may comment the lines above and use the following:
                    # s1 = sample["s1"].to(device)
                    # img = {"s1": s1}

                    # load target
                    y = sample[train_config['target']].long().to(device)

                    # model output
                    y_hat = model(img)

                    # loss computation
                    loss = criterion(y_hat, y)

                    # get prediction 
                    _, pred = torch.max(y_hat, dim=1)

                    epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
                    metrics.add_batch(y, pred)


                    pbar.set_description(f"Validation Loss:{epoch_losses[-100:].mean():.4}")

                mean_loss = epoch_losses.mean()

                val_stats = {
                    "validation_loss": mean_loss.item(),
                    "validation_average_accuracy": metrics.get_average_accuracy(),
                    "validation_overall_accuracy": metrics.get_overall_accuracy(),
                    **{
                        "validation_accuracy_" + k: v
                        for k, v in metrics.get_classwise_accuracy().items()
                    },
                }

                print(f"Epoch:{epoch}", val_stats)
                
                # Save model checkpoint every 2 epochs 
                if epoch % 2 == 0:
                    if epoch == 0:
                        continue

                    save_weights_path = (
                        "checkpoints/" + "-".join(["classifier", "epoch", str(epoch)]) + ".pth"
                    )
                    torch.save(model.state_dict(), save_weights_path)

if __name__ == '__main__':
    # This is necessary for multiprocessing support.
    # You can omit freeze_support() if you're not freezing your script with tools like PyInstaller or cx_Freeze
    # multiprocessing.freeze_support() 
    main()