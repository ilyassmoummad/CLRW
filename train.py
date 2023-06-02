import torch
from utils import adjust_learning_rate
from args import args
if args.wandb:
    import wandb

def cifar10_train_backbone_epoch(train_loader, backbone, criterion, optimizer):
    """one epoch training"""
    backbone.train()
    epoch_loss = 0.0

    for idx, (images, labels) in enumerate(train_loader):
        
        x1 = images[0].to(args.device)
        x2 = images[1].to(args.device)
        labels = labels.to(args.device)
        
        # compute loss
        _, p1 = backbone(x1); _, p2 = backbone(x2)
        loss = criterion(p1, p2)
        epoch_loss += loss.item()
        
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = epoch_loss/len(train_loader)

    # print info
    print(f"Current Train Loss : {format(epoch_loss, '.4f')}")
    return epoch_loss

def cifar10_train_backbone(train_loader, backbone, criterion, optimizer, epochs):
    for i in range(1, epochs+1):
        adjust_learning_rate(optimizer, i, args)
        print(f"Epoch {i}")
        epoch_loss = cifar10_train_backbone_epoch(train_loader, backbone, criterion, optimizer)
        if args.wandb:
            wandb.log({'pretrain_loss': epoch_loss})

    return backbone