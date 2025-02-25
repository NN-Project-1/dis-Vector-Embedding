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
        
        trainer = Trainer(data_loader, args, config)
        trainer.train()
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--resume_iterations', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--checkpoint_save_epoch', type=int, default=100)
    parser.add_argument('--checkpoint_save_interval', type=int, default=100)
    parser.add_argument('--stage', type=int, default=1, help='0: preprocessing; 1: training')
    parser.add_argument('--config', type=str, default='Large_B')
    parser.add_argument('--mode', type=str, default='Training') 
    args = parser.parse_args()

    config = yaml.safe_load(open(os.path.join('configs', f'{args.config}.yaml'), 'r'))
    config = Dict2Class(config)
    
    if args.mode == 'F':
        config.mode = 'F'
        config.pitch_dim = config.content_dim + config.pitch_dim 
    
    main(config, args)
