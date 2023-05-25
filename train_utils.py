import torch
from utils import lin_adjust_learning_rate
from args import args
import wandb

#CIFAR10

def cifar10_train_linear_epoch(train_loader, backbone, classifier, criterion, optimizer):

    backbone.eval()
    classifier.train()

    correct = 0
    total = 0
    epoch_loss = 0.

    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        bsz = labels.shape[0]

        # compute loss and accuracy
        with torch.no_grad():
            embeds, _ = backbone(images)
        output = classifier(embeds)
        _, labels_predicted = torch.max(output, dim=1)
        correct += (labels_predicted==labels).sum()
        total += labels.size(0)
        loss = criterion(output, labels)
        epoch_loss += loss.item()

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = correct/total
    epoch_loss = epoch_loss / len(train_loader)
    #print(f"Train Loss: {format(epoch_loss, '.4f')}\tTrain Acc: {format(acc, '.4f')}")
    return acc, epoch_loss
    
    
def cifar10_val_linear_epoch(val_loader, backbone, classifier, criterion):

    backbone.eval()
    classifier.eval()

    correct = 0
    total = 0
    epoch_loss = 0.

    for idx, (images, labels) in enumerate(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        bsz = labels.shape[0]

        # compute loss and accuracy
        with torch.no_grad():
            embeds, _ = backbone(images)
            output = classifier(embeds)
        _, labels_predicted = torch.max(output, dim=1)
        correct += (labels_predicted==labels).sum()
        total += labels.size(0)
        loss = criterion(output, labels)
        epoch_loss += loss.item()

    acc = correct/total
    epoch_loss = epoch_loss / len(val_loader)
    #print(f"Val Loss: {format(epoch_loss, '.4f')}\tVal Acc: {format(acc, '.4f')}")
    return acc, epoch_loss

def cifar10_train_linear(train_loader, val_loader, backbone, classifier, criterion, optimizer, epochs):
    
    best_acc = None
    best_epoch = 0

    print(f"linear classification")

    for i in range(1, epochs+1):
        lin_adjust_learning_rate(optimizer, i, args)
        print(f"epoch {i}")
        train_acc, train_loss = cifar10_train_linear_epoch(train_loader, backbone, classifier, criterion, optimizer)
        print(f"Train Loss: {format(train_loss, '.4f')}\tTrain Acc: {format(train_acc, '.4f')}")
        val_acc, val_loss = cifar10_val_linear_epoch(val_loader, backbone, classifier, criterion)
        print(f"Val Loss: {format(val_loss, '.4f')}\tVal Acc: {format(val_acc, '.4f')}")

        if args.wandb:
            wandb.log({'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss})

        if best_acc is None:
            best_acc = val_acc

        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = i

    print(f"Best Val Acc: {format(best_acc, '.4f')} at epoch {best_epoch}")
    if args.wandb:
        wandb.log({'best_val_acc': best_acc})
    return backbone, classifier, best_acc