import Initialize
import Model
from Dataset import EBertDataset
import Trainer
import  Evaluator
from utils import make_data, get_logger

import argparse
import datetime
import torch.cuda
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adadelta

# initialize config
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, help='File path to initailize model para')
parser.add_argument('--data_dir', type=str, help="path to store data")
parser.add_argument('--save_path', type=str, help="path to save model and weight")
parser.add_argument('--log_folder', type=str, default='/data/yujie/Ebert/log/')
parser.add_argument('--eval', type=bool, default=False, help='whether do evaluation')
parser.add_argument('--shuffle_data', type=bool, default=False, help='whether shuffle the dataset')
args = parser.parse_args()
with open(args.config_path, 'r') as file:
    lines = file.readlines()

# initialize logger
log_path = args.log_folder + f'log_{start_time}.log'
logger = get_logger(log_path)

# save model paras in dict
paras = {}
for line in lines:
    key, value = line.strip().split('=')
    paras[key] = float(value) if '.' in value else int(value)

# initialize config
config = Initialize.Config(paras)

# initialize device
initial_device = Initialize.initailDevice()
device_index = str(initial_device.chosen_gpu)
device = 'cuda:'+device_index
device = torch.device(device)

# initialize model
model = Model.Bert(config, device)
model.to(device)

# initialize datasetloader
trainset, valset, testset=make_data(args.data_dir, shuffle=False, start_time=start_time)
Trainset = EBertDataset(dataset=trainset)
Valset=EBertDataset(dataset=valset)
Testset=EBertDataset(dataset=testset)

Trainloader = DataLoader(Trainset, batch_size=config.batch_size, shuffle=True)
Valloader=DataLoader(Valset, batch_size=config.batch_size, shuffle=True)
Testloader=DataLoader(Testset, batch_size=config.batch_size, shuffle=True)

# training
ebert_trainer = Trainer.BasicTrainer(config, Trainloader=Trainloader, Valloader=Valloader
                                     , device=device, model=model, criterion=nn.CrossEntropyLoss(),
                                     optimizer = Adadelta,save_path=args.save_path, lr = config.lr)
ebert_trainer.training(start_time=start_time)

#evaluation
if config.eval == True:
    ebert_evaluator = Evaluator.BasicEvaluator(config=config, dataloader=Testloader)
    ebert_evaluator.evaluate()


