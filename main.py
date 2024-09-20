import os
import yaml
import argparse
from torch.backends import cudnn

from trainer_new_iter import Trainer
from data_loader import get_loader
from data_preprocessing import preprocess_data
from utils import Dict2Class

def main(config, args):
    cudnn.benchmark = True

    if args.stage == 0:
        preprocess_data(config)
    elif args.stage == 1:
        data_loader = get_loader(config)
        
        solver = Trainer(data_loader, args, config)
        solver.train()
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_iters', type=int, default=10000)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--resume_iters', type=int, default=0)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--ckpt_save_epoch', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=100)
    parser.add_argument('--stage', type=int, default=1, help='0: preprocessing; 1: training')
    parser.add_argument('--config_name', type=str, default='spsp2-large')
    parser.add_argument('--model_type', type=str, default='Training') 
    args = parser.parse_args()

    config = yaml.safe_load(open(os.path.join('configs', f'{args.config_name}.yaml'), 'r'))
    config = Dict2Class(config)
    if args.model_type == 'F':
        config.model_type = 'F'
        config.dim_pit = config.dim_con + config.dim_pit 
    
    main(config, args)
