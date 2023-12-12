# Import needed packages
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from distutils.util import strtobool
from tqdm import tqdm
from torchvision.models import resnet18, resnet50
import os


def log_metrics(epoch, phase, metrics, log_file="training_log_cacao_batch.txt"):
    """
    Logs the metrics to a specified file.

    :param epoch: Current epoch number.
    :param phase: 'Training' or 'Validation'.
    :param metrics: Dictionary containing metrics like loss and accuracy.
    :param log_file: File path to save the log.
    """
    with open(log_file, "a") as file:
        log_entry = f"Epoch {epoch} [{phase}]:\n"
        for key, value in metrics.items():
            log_entry += f"    {key}: {value}\n"
        file.write(log_entry + "\n")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # Data configurations:
    data_config = {
        'train_dir': '../data', # path to the training directory,  
        'val_dir': '../data', # path to the validation directory,
        'train_mode': 'validation', # can be one of the following: 'test', 'validation'
        'val_mode': 'test', # can be one of the following: 'test', 'validation'
        'num_classes': 9, # number of classes in the dataset.
        'clip_sample_values': True, # clip (limit) values
        'train_used_data_fraction': 1, # fraction of data to use, should be in the range [0, 1]
        'val_used_data_fraction': 1,
        'image_px_size': 224, # image size (224x224)
        'cover_all_parts_train': True, # if True, if image_px_size is not 224 during training, we use a random crop of the image
        'cover_all_parts_validation': True, # if True, if image_px_size is not 224 during validation, we use a non-overlapping sliding window to cover the entire image
        'seed': 42,
    }

    # Ensure deterministic behavior
    random.seed(data_config['seed'])
    np.random.seed(data_config['seed'])
    torch.manual_seed(data_config['seed'])
    torch.cuda.manual_seed_all(data_config['seed'])

    from dfc_dataset import DFCDataset

    # Create Training Dataset
    train_dataset = DFCDataset(
        data_config['train_dir'],
        mode=data_config['train_mode'],
        clip_sample_values=data_config['clip_sample_values'],
        used_data_fraction=data_config['train_used_data_fraction'],
        image_px_size=data_config['image_px_size'],
        cover_all_parts=data_config['cover_all_parts_train'],
        seed=data_config['seed'],
        add_cacao=True,
    )

    # Create Validation Dataset
    val_dataset = DFCDataset(
        data_config['val_dir'],
        mode=data_config['val_mode'],
        clip_sample_values=data_config['clip_sample_values'],
        used_data_fraction=data_config['val_used_data_fraction'],
        image_px_size=data_config['image_px_size'],
        cover_all_parts=data_config['cover_all_parts_validation'],
        seed=data_config['seed'],
        add_cacao=True
    )

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
        'epochs': 201, 
        'target': 'dfc_label'
    }

    # path to the checkpoint
    checkpoint = torch.load(
        "../checkpoints/resnet50.pth"
    ) 

    from utils import save_checkpoint_single_model, dotdictify

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu:0")
    print(device)

    import torchvision.models as models

    class DoubleResNetSimCLRDownstream(torch.nn.Module):
        """concatenate outputs from two backbones and add one linear layer"""

        def __init__(self, base_model, out_dim):
            super(DoubleResNetSimCLRDownstream, self).__init__()

            self.resnet_dict = {"resnet18": models.resnet18,
                                "resnet50": models.resnet50,}
            

            self.backbone2 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
            dim_mlp2 = self.backbone2.fc.in_features
            
            # If you are using multimodal data you can un-comment the following lines:
            # self.backbone1 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
            # dim_mlp1 = self.backbone1.fc.in_features
            
            # add final linear layer
            self.fc = torch.nn.Linear(dim_mlp2, out_dim, bias=True)
            # self.fc = torch.nn.Linear(dim_mlp1 + dim_mlp2, out_dim, bias=True)

            # self.backbone1.fc = torch.nn.Identity()
            self.backbone2.fc = torch.nn.Identity()

        def _get_basemodel(self, model_name):
            try:
                model = self.resnet_dict[model_name]
            except KeyError:
                raise InvalidBackboneError(
                    "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
            else:
                return model

        def forward(self, x):
            x2 = self.backbone2(x["s2"])

            # If you are using multimodal data you can un-comment the following lines and comment z = self.fc(x2):
            # x1 = self.backbone1(x["s1"])
            # z = torch.cat([x1, x2], dim=1)
            # z = self.fc(z)
        
            z = self.fc(x2)
            
            return z
        
        def load_trained_state_dict(self, weights):
            """load the pre-trained backbone weights"""
            
            # remove the MLP projection heads
            for k in list(weights.keys()):
                if k.startswith(('backbone1.fc', 'backbone2.fc')):
                    del weights[k]
            
            log = self.load_state_dict(weights, strict=False)
            assert log.missing_keys == ['fc.weight', 'fc.bias']
            
            # freeze all layers but the last fc
            for name, param in self.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False

    base_model = "resnet50"
    num_classes = 9

    model = eval('DoubleResNetSimCLRDownstream')(base_model, num_classes)

    model.backbone2.conv1 = torch.nn.Conv2d(
        train_config['s2_input_channels'],
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )

    model.load_trained_state_dict(checkpoint["state_dict"])
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
                if torch.isnan(sample["s2"]).any():
                    # some s1 scenes are known to have NaNs...
                    continue
            
            # load input
            s2 = sample["s2"].to(device)
            img = {"s2": s2}
            
            # if you are using a unimodal dataset (s1 for example), you may un-comment the following lines:
            # s1 = sample["s1"].to(device)
            # img = {"s1": s1, "s2": s2}
            
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
        log_metrics(epoch, "Training", train_stats)

        if epoch % 5 == 0:  

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
                        if torch.isnan(sample["s2"]).any():
                            # some s1 scenes are known to have NaNs...
                            continue
                    # load input
                    s2 = sample["s2"].to(device)
                    img = {"s2": s2}

                    # if you are using a unimodal dataset (s1 for example), you may un-comment the following lines:
                    # s1 = sample["s1"].to(device)
                    # img = {"s1": s1, "s2": s2}

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
                log_metrics(epoch, "Validation", val_stats)
                print(f"Epoch:{epoch}", val_stats)
                
                # Save model checkpoint every 2 epochs 
                if epoch % 5 == 0:
                    if epoch == 0:
                        continue

                    save_weights_path = (
                        "checkpoints/cacao_batch/" + "-".join(["classifier", "epoch", str(epoch)]) + ".pth"
                    )
                    torch.save(model.state_dict(), save_weights_path)
