import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from collections import OrderedDict
from utils import quantize_f0_torch
from model import Vector_Mode as ModelGenerator
from model import InterpLnr


class ModelTrainer:

    def __init__(self, data_loader, args, config):
        self.args = args
        self.epochs = self.args.num_epochs
        self.start_epoch = self.args.resume_epoch
        self.log_interval = self.args.log_step
        self.save_checkpoint_interval = self.args.ckpt_save_epoch
        self.config = config
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)
        self.learning_rate = self.config.lr
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.experiment_name = self.config.experiment
        self.bottleneck_size = self.config.bottleneck
        self.training_mode = self.config.model_type
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.config.device_id}' if self.use_cuda else 'cpu')

        self.checkpoint_directory = self.config.model_save_dir
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.initialize_model()
        self.best_loss_step = 0
        self.best_loss = float('inf')

    def initialize_model(self):
        self.model = ModelGenerator(self.config)
        self.display_model_summary(self.model, self.training_mode)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.interpolator = InterpLnr(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2), weight_decay=1e-6
        )

    def load_checkpoint(self, resume_epoch):
        print(f'Loading trained model from epoch {resume_epoch}...')
        checkpoint_filename = f'{self.experiment_name}-{self.bottleneck_size}-{self.training_mode}-{resume_epoch}.ckpt'
        checkpoint_path = os.path.join(self.checkpoint_directory, checkpoint_filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for key, value in checkpoint['model'].items():
                new_state_dict[key[7:]] = value  # Remove 'module.' prefix for DataParallel models
            self.model.load_state_dict(new_state_dict)

        self.learning_rate = self.optimizer.param_groups[0]['lr']

    def train(self):
        start_epoch = self.start_epoch or 0

        if start_epoch:
            print('Resuming training...')
            self.epochs += start_epoch
            self.load_checkpoint(start_epoch)

        print(f'Current learning rate: {self.learning_rate}.')
        print('Starting training...')
        start_time = time.time()
        self.model.train()

        for epoch in range(start_epoch, self.epochs):
            running_loss = 0.0
            epoch_start_time = datetime.datetime.now()

            try:
                (speaker_id, target_spectrogram, rhythm_data, content_data, pitch_data, 
                 timbre_data, crop_length, timbre_features) = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.data_loader)
                (speaker_id, target_spectrogram, rhythm_data, content_data, pitch_data, 
                 timbre_data, crop_length, timbre_features) = next(self.data_iterator)

            target_spectrogram = target_spectrogram.to(self.device)
            rhythm_data = rhythm_data.to(self.device)
            content_data = content_data.to(self.device)
            pitch_data = pitch_data.to(self.device)
            timbre_features = timbre_features.to(self.device)
            timbre_data = timbre_data.to(self.device)
            crop_length = crop_length.to(self.device)

            content_pitch_data = torch.cat((content_data, pitch_data), dim=-1)
            content_pitch_data_interpolated = self.interpolator(content_pitch_data, crop_length)
            pitch_data_interpolated = quantize_f0_torch(content_pitch_data_interpolated[:, :, -1])[0]
            content_pitch_data_interpolated = torch.cat(
                (content_pitch_data_interpolated[:, :, :-1], pitch_data_interpolated), dim=-1
            )

            generated_spectrogram = self.model(
                content_pitch_data_interpolated, rhythm_data, timbre_data, timbre_features
            )
            identity_loss = F.mse_loss(generated_spectrogram, target_spectrogram)
            print(f"Loss details: {identity_loss.item():.6f}")

            loss = identity_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (epoch + 1) % self.log_interval == 0:
                print(f'Epoch: {epoch + 1}, Loss: {running_loss / self.log_interval:.3f}')
                running_loss = 0.0

            epoch_end_time = datetime.datetime.now()
            print(f"Epoch {epoch + 1} Duration: {epoch_end_time - epoch_start_time}")

            if (epoch + 1) % self.save_checkpoint_interval == 0:
                checkpoint_filename = f'{self.experiment_name}-{self.bottleneck_size}-{self.training_mode}-{epoch+1}.ckpt'
                checkpoint_path = os.path.join(self.checkpoint_directory, checkpoint_filename)
                torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, checkpoint_path)
                print(f'Saved model checkpoint at {checkpoint_path}.')

        print('Training Complete')
