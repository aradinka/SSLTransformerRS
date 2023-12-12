import json
from tqdm import tqdm
import torch
import torch.nn.functional as F

from dfc_dataset import DFCDataset
from Transformer_SSL.models import build_model
from Transformer_SSL.models.swin_transformer import DoubleSwinTransformerSegmentation
from utils import dotdictify
from metrics import PixelwiseMetrics



FINETUNE = False
EPOCHS = 200
TARGET = "dfc"
TRAIN_DIR = "data"
VAL_DIR = "data"
PRETRAINED_PATH = "checkpoints/swin_t.pth"
BACKBONE_CONFIG = "configs/backbone_config.json"
DEVICE = "cuda"

train_config = {
    's1_input_channels': 2,
    's2_input_channels': 13,
    'finetuning': FINETUNE, # If false, backbone layers is frozen and only the head is trained
    'classifier_lr': 3e-6,
    'learning_rate': 0.00001,
    'adam_betas': (0.9, 0.999), 
    'weight_decay': 0.001,
    'dataloader_workers': 4,
    'batch_size': 16,
    'epochs': EPOCHS, 
    'target': TARGET,
}

data_config = {
    'train_dir': TRAIN_DIR, # path to the training directory,  
    'val_dir': VAL_DIR, # path to the validation directory,
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

### Load Pretrain ###
checkpoint = torch.load(PRETRAINED_PATH, map_location=torch.device(DEVICE)) 
weights = checkpoint["state_dict"]

s1_weights = {k[len("backbone1."):]: v for k, v in weights.items() if "backbone1" in k}
s2_weights = {k[len("backbone2."):]: v for k, v in weights.items() if "backbone2" in k}

with open(BACKBONE_CONFIG, "r") as fp:
    swin_conf = dotdictify(json.load(fp))

s1_backbone = build_model(swin_conf.model_config)
swin_conf.model_config.MODEL.SWIN.IN_CHANS = 13
s2_backbone = build_model(swin_conf.model_config)

s1_backbone.load_state_dict(s1_weights)
s2_backbone.load_state_dict(s2_weights)

device = torch.device(DEVICE)
model = DoubleSwinTransformerSegmentation(
        s1_backbone, s2_backbone, out_dim=data_config['num_classes'], device=device
    )
model = model.to(device)

### Training ###
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

criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
optimizer = torch.optim.Adam(
    parameters,
    lr=train_config['learning_rate'],
    betas=train_config['adam_betas'],
    weight_decay=train_config['weight_decay'],
)


step = 0
for epoch in range(train_config['epochs']):
    # Model Training
    model.train()
    step += 1

    pbar = tqdm(train_loader)

    # track performance
    epoch_losses = torch.Tensor()
    metrics = PixelwiseMetrics(data_config['num_classes'])
    

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
        y = sample[train_config['target']].squeeze().type(torch.LongTensor).to(device)
        
        # model output
        y_hat = model(img)
        
        # loss computation
        loss = criterion(y_hat, y)
        
        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # get prediction 
        probas = F.softmax(y_hat, dim=1)
        pred = torch.argmax(probas, axis=1)

        epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
        metrics.add_batch(y, pred)

        pbar.set_description(f"Epoch:{epoch}, Training Loss:{epoch_losses[-100:].mean():.4}")

    mean_loss = epoch_losses.mean()

    train_stats = {
            "train_loss": mean_loss.item(),
            "train_average_accuracy": metrics.get_average_accuracy(),
            **{
                "train_accuracy_" + k: v
                for k, v in metrics.get_classwise_accuracy().items()
            },
        }
    print(train_stats)

    if epoch % 5 == 0:  

        # Model Validation
        model.eval()
        pbar = tqdm(val_loader)

        # track performance
        epoch_losses = torch.Tensor()
        metrics = PixelwiseMetrics(data_config['num_classes'])

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
            if epoch % 5 == 0:
                if epoch == 0:
                    continue

                save_weights_path = (
                    "checkpoints/" + "-".join(["segmentation", "epoch", str(epoch)]) + ".pth"
                )
                torch.save(model.state_dict(), save_weights_path)