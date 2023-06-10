import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--wandb", action='store_true') #use for wandb
parser.add_argument("--simclr", action='store_true') #use SimCLR instead of our proposed CLRW
parser.add_argument("--autoaugment", action='store_true') #use Torchvision's AutoAugment as DA
parser.add_argument("--entity", type=str, default='') #entity name for wandb
parser.add_argument("--datadir", type=str, default='.') #path to save downloaded dataset
parser.add_argument("--dataset", type=str, default='cifar10') #dataset to train on ['cifar10']
parser.add_argument("--nworkers", type=int, default=4) #number of workers for the dataloader
parser.add_argument("--model", type=str, default='resnet18') #model to train, check model_dict in models.py
parser.add_argument("--projname", type=str, default='') #project name for wandb ['GSP23','sandbox']
parser.add_argument("--tmpdir", type=str, default='.') #tmp dir
parser.add_argument("--device", type=str, default='cuda:0') #device to train on
parser.add_argument("--pin_memory", action='store_true') #use for wandb
parser.add_argument("--bs", type=int, default=256) #batch size for representation learning
parser.add_argument("--bs2", type=int, default=256) #batch size for lincls
parser.add_argument("--wd", type=float, default=5e-4) #weight decay #avant : 1e-4
parser.add_argument("--momentum", type=float, default=0.9) #sgd momentum
parser.add_argument("--lr", type=float, default=6e-2) #learning rate #avant : 5e-2
parser.add_argument("--lr2", type=float, default=1e-1) #learning rate for linear eval
parser.add_argument("--epochs", type=int, default=800) #nb of epochs of ssl
parser.add_argument("--epochs2", type=int, default=100) #nb of epochs of linear classif
parser.add_argument("--tau", type=float, default=1.0) #similarity temperature
parser.add_argument("--schedule", type=int, default=[60, 80], nargs='*') #lr schedule for lin cls

## GCN arguments 
parser.add_argument("--gcn", action='store_true') #Use GCN model instead of SimCLR
parser.add_argument("--nfm", type=float, default=0) # probability of masking a node feature
parser.add_argument("--edgem", type=float, default=0) # probability of masking an edge

args = parser.parse_args()