import os 
import torch
import pickle  
import numpy as np
import librosa

from torch.utils import data
from torch.utils.data.sampler import Sampler

from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

class WavDataset(data.Dataset):

    def __init__(self, config):
        self.feature_directory = config.feat_dir
        self.wav_directory = os.path.join(self.feature_directory, config.wav_dir)
        print('Loading data...')

        metadata_path = os.path.join(self.feature_directory, 'dataset.pkl')
        metadata = pickle.load(open(metadata_path, "rb"))
        
        dataset_entries = [None] * len(metadata)
        self.load_data(metadata, dataset_entries)
        self.dataset = list(dataset_entries)
        self.total_samples = len(self.dataset)

    def load_data(self, metadata, dataset_entries):  
        for index, entry in enumerate(metadata):    
            utterances = len(entry) * [None]
            utterances[0] = entry[0]
            utterances[1] = entry[1]
            wav_data = np.load(os.path.join(self.wav_directory, entry[2])) 
            utterances[2] = (wav_data,)
            dataset_entries[index] = utterances  
        
    def __getitem__(self, index):
        utterance_data = self.dataset[index]
        speaker_id = utterance_data[0]
        speaker_embedding = utterance_data[1]
        wav_data = utterance_data[2][0]
        
        spectral_envelope = get_spenv(wav_data)
        mel_spectrogram = get_spmel(wav_data)
        timbre_characteristics = extract_timbre_features(wav_data, 16000)
        
        return wav_data, speaker_id, mel_spectrogram, spectral_envelope, mel_spectrogram, speaker_embedding, timbre_characteristics

    def __len__(self):
        return self.total_samples

class DataCollator(object):
    def __init__(self, config):
        self.min_sequence_length = config.min_len_seq
        self.max_sequence_length = config.max_len_seq
        self.max_padding_length = config.max_len_pad

    def __call__(self, batch):
        processed_batch = []
        for sample in batch:
            _, speaker_id, mel_target, spectral_envelope, mel_spectrogram, speaker_embedding, timbre_characteristics = sample
            crop_length = np.random.randint(self.min_sequence_length, self.max_sequence_length + 1)
            start_idx = np.random.randint(0, len(mel_target) - crop_length)

            mel_target = mel_target[start_idx:start_idx + crop_length, :]  
            spectral_envelope = spectral_envelope[start_idx:start_idx + crop_length, :]
            mel_spectrogram = mel_spectrogram[start_idx:start_idx + crop_length, :]
            timbre_characteristics = timbre_characteristics[:, start_idx:start_idx + crop_length]

            mel_target = np.clip(mel_target, 0, 1)
            spectral_envelope = np.clip(spectral_envelope, 0, 1)
            mel_spectrogram = np.clip(mel_spectrogram, 0, 1)

            mel_target = np.pad(mel_target, ((0, self.max_padding_length - mel_target.shape[0]), (0, 0)), 'constant')
            spectral_envelope = np.pad(spectral_envelope, ((0, self.max_padding_length - spectral_envelope.shape[0]), (0, 0)), 'constant')
            mel_spectrogram = np.pad(mel_spectrogram, ((0, self.max_padding_length - mel_spectrogram.shape[0]), (0, 0)), 'constant')
            timbre_characteristics = np.pad(timbre_characteristics, ((0, 0), (0, self.max_padding_length - timbre_characteristics.shape[1])), 'constant')

            processed_batch.append((speaker_id, mel_target, spectral_envelope, mel_spectrogram, speaker_embedding, crop_length, timbre_characteristics))

        speaker_id, mel_target, spectral_envelope, mel_spectrogram, speaker_embedding, crop_length, timbre_characteristics = zip(*processed_batch)
        speaker_id = list(speaker_id)
        mel_target = torch.FloatTensor(np.stack(mel_target, axis=0))
        spectral_envelope = torch.FloatTensor(np.stack(spectral_envelope, axis=0))
        mel_spectrogram = torch.FloatTensor(np.stack(mel_spectrogram, axis=0))
        speaker_embedding = torch.FloatTensor(np.stack(speaker_embedding, axis=0))
        crop_length = torch.LongTensor(np.stack(crop_length, axis=0))
        timbre_characteristics = torch.FloatTensor(np.stack(timbre_characteristics, axis=0)).permute(0, 2, 1)  

        return speaker_id, mel_target, spectral_envelope, mel_spectrogram, speaker_embedding, crop_length, timbre_characteristics


def get_data_loader(config):
    dataset = WavDataset(config)
    collator = DataCollator(config)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  drop_last=False,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=collator)

    return data_loader
