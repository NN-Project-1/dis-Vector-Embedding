
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from collections import OrderedDict
from utils import quantize_f0_torch
import datetime
from model import Vector_Mode as Generator
from model import InterpLnr


class Trainer(object):

    def __init__(self, data_loader, args, config):

        self.args = args
        self.num_epochs = self.args.num_epochs
        self.resume_epoch = self.args.resume_epoch
        self.log_step = self.args.log_step
        self.ckpt_save_epoch = self.args.ckpt_save_epoch
        self.config = config
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.lr = self.config.lr
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.experiment = self.config.experiment
        self.bottleneck = self.config.bottleneck
        self.model_type = self.config.model_type
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(self.config.device_id) if self.use_cuda else 'cpu')

        self.model_save_dir = self.config.model_save_dir
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.build_model()
        self.min_loss_step = 0
        self.min_loss = float('inf')

    def build_model(self):        
        self.model = Generator(self.config) 
        self.print_network(self.model, self.model_type)
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.Interp = InterpLnr(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=1e-6)
        self.Interp.to(self.device)


    def restore_model(self, resume_epoch):
        print('Loading the trained models from epoch {}...'.format(resume_epoch))
        ckpt_name = f'{self.experiment}-{self.bottleneck}-{self.model_type}-{resume_epoch}.ckpt'
        ckpt = torch.load(os.path.join(self.model_save_dir, ckpt_name), map_location=lambda storage, loc: storage)
        try:
            self.model.load_state_dict(ckpt['model'])
        except:
            new_state_dict = OrderedDict()
            for k, v in ckpt['model'].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)
        self.lr = self.optimizer.param_groups[0]['lr']

    def train(self):
        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch:
            print('Resuming ...')
            start_epoch = self.resume_epoch
            self.num_epochs += self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Learning rate cache for decaying.
        lr = self.lr
        print ('Current learning rates, lr: {}.'.format(lr))

        # Start training.
        print('Start training...')
        start_time = time.time()
        self.model = self.model.train()
        for epoch in range(start_epoch, self.num_epochs):
            running_loss = 0.0
            epoch_start_time = datetime.datetime.now()

            # Load data
            try:
               spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop, timbre_features = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop, timbre_features = next(self.data_iter)

    
            #Train the model
            if self.model_type == 'Training':
                # Move data to GPU if available
                spmel_gt = spmel_gt.to(self.device)
                rhythm_input = rhythm_input.to(self.device)
                content_input = content_input.to(self.device)
                pitch_input = pitch_input.to(self.device)
                timbre_features = timbre_features.to(self.device)
                timbre_input=timbre_input.to(self.device)
                len_crop = len_crop.to(self.device)

                # Prepare input data and apply random resampling
                content_pitch_input = torch.cat((content_input, pitch_input), dim=-1) # [B, T, F+1]
                content_pitch_input_intrp = self.Interp(content_pitch_input, len_crop) # [B, T, F+1]
                pitch_input_intrp = quantize_f0_torch(content_pitch_input_intrp[:, :, -1])[0] # [B, T, 257]
                content_pitch_input_intrp = torch.cat((content_pitch_input_intrp[:,:,:-1], pitch_input_intrp), dim=-1) # [B, T, F+257]
                
                # Identity mapping loss
                spmel_output = self.model(content_pitch_input_intrp, rhythm_input, timbre_input ,timbre_features)
                loss_id = F.mse_loss(spmel_output, spmel_gt)
                print("Loss details :",loss_id)

            elif self.model_type == 'F':
                # Move data to GPU if available
                rhythm_input = rhythm_input.to(self.device)
                pitch_input = pitch_input.to(self.device)
                len_crop = len_crop.to(self.device)

                # Prepare input data and apply random resampling
                pitch_gt = quantize_f0_torch(pitch_input)[1].view(-1)
                content_input = content_input.to(self.device)
                content_pitch_input = torch.cat((content_input, pitch_input), dim=-1) # [B, T, F+1]
                content_pitch_input = self.Interp(content_pitch_input, len_crop) # [B, T, F+1]
                pitch_input_intrp = quantize_f0_torch(content_pitch_input[:, :, -1])[0] # [B, T, 257]
                pitch_input = torch.cat((content_pitch_input[:,:,:-1], pitch_input_intrp), dim=-1) # [B, T, F+257]
                
                # Identity mapping loss
                pitch_output = self.model(rhythm_input, pitch_input).view(-1, self.config.dim_f0)
                loss_id = F.cross_entropy(pitch_output, pitch_gt)

            else:
                raise ValueError

            # Backward and optimize.
            loss = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if epoch % self.log_step == self.log_step - 1:  # Print every log_step mini-batches
                print('epoch: {}, loss: {:.3f}'.format(epoch + 1, running_loss / self.log_step))
                running_loss = 0.0

            # Save model checkpoints
            epoch_end_time = datetime.datetime.now()
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch + 1} took: {epoch_elapsed_time}")
            if (epoch + 1) % self.ckpt_save_epoch == 0:
                ckpt_name = f'{self.experiment}-{self.bottleneck}-{self.model_type}-{epoch+1}.ckpt'
                torch.save({
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                          }, os.path.join(self.model_save_dir, ckpt_name))
                print('Saving model checkpoint into {}...'.format(self.model_save_dir))

        print('Finished Training')
