import torch
from torch import nn
from torchvision import datasets, transforms
from utils import TwoCropTransform
from models import LinearClassifier
from train import cifar10_train_backbone
from train_utils import cifar10_train_linear
from losses import RandomWalkLoss
from args import args
from models import RWResNet, model_dict
import os
if args.wandb:
        import wandb

OUT_DIM = model_dict[args.model][-1]
NUM_CLASSES = 10 if args.dataset == 'cifar10' else None

os.makedirs(args.datadir, exist_ok=True)
os.makedirs(args.tmpdir, exist_ok=True)

if args.autoaugment:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),    
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                transforms.RandomErasing(0.1)
        ])
else:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),      
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

train_dataset = datasets.CIFAR10(root=args.datadir,
                        transform=TwoCropTransform(train_transform),
                        train=True,
                        download=True)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=args.nworkers, pin_memory=True, drop_last=True)

backbone = RWResNet(name=args.model, head='mlp', out_dim=OUT_DIM).to(args.device)
classifier = LinearClassifier(name=args.model, num_classes=NUM_CLASSES).to(args.device)

optim_params = list(backbone.parameters())

optimizer_ssl = torch.optim.SGD(optim_params, lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.wd)

if args.simclr:
        criterion_ssl = SupConLoss(args.tau)
else:
        criterion_ssl = RandomWalkLoss(args.tau)

optimizer_linear = torch.optim.SGD(classifier.parameters(),
                          lr=args.lr2,
                          momentum=args.momentum,)

criterion_linear = nn.CrossEntropyLoss()

if args.wandb:
        os.makedirs(args.tmpdir, exist_ok=True)
        wandb.init(project=args.projname, entity=args.entity, dir=args.tmpdir) #['GSP', 'sandbox']
        wandb.config.update(args)

backbone = cifar10_train_backbone(train_loader, backbone, criterion_ssl, optimizer_ssl, args.epochs)

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0),
                                     ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
val_transform = transforms.Compose([
        transforms.Resize(int(32 * (8 / 7))),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

train_dataset = datasets.CIFAR10(root=args.datadir,
                        transform=train_transform,
                        train=True,
                        download=True)
val_dataset = datasets.CIFAR10(root=args.datadir,
                        transform=val_transform,
                        train=False,
                        download=True)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs2, shuffle=True,
        num_workers=args.nworkers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs2, shuffle=False,
        num_workers=args.nworkers, pin_memory=True)

backbone, classifier, acc = cifar10_train_linear(train_loader, val_loader, backbone, classifier, criterion_linear, optimizer_linear, args.epochs2)
