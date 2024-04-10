import os
import yaml
from collections import OrderedDict
import torch
import numpy as np
from soundfile import read, write
from Dis_vector_model.Dis_models import Generator_3 as Generator
from Dis_vector_model.Dis_models import Generator_6 as F0_Converter
from Dis_vector_model.utils import *

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(ckpt['model'])
    except:
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)

def pad_fea(fea):
    if len(fea) >= T:
        return fea[:T]
    else:
        return np.pad(fea, ((0, T - len(fea)), (0, 0)), 'constant')

def create_feats(wav, gen, spk_id, config):
    lo, hi = 50, 250  # Default values

    if gen == 'M':
        lo, hi = 50, 250
    else:
        lo, hi = 100, 600

    if wav.shape[0] % 256 == 0:
        wav = np.concatenate((wav, np.array([1e-06])), axis=0)
    _, f0_norm = extract_f0(wav, fs, lo, hi)
    f0, sp, ap = get_world_params(wav, fs)
    f0 = average_f0s([f0])[0]
    wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs)

    rhy_input = pad_fea(get_spenv(wav_mono))
    con_input = pad_fea(get_spmel(wav_mono))
    pit_input = pad_fea(quantize_f0_numpy(f0_norm)[0])
    tim_input = np.zeros((82,), dtype=np.float32)
    tim_input[int(spk_id)] = 1.0

    return (torch.FloatTensor(x).unsqueeze(0).to(device) for x in (rhy_input, con_input, pit_input, tim_input))

config = yaml.safe_load(open(f'/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/Dis_vector_model/configs/model_config.yaml', 'r'))
config = Dict2Class(config)
config.train = False

T = 192 # maximum number of frames in the output mel-spectrogram
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fs = 16000

G = Generator(config).eval().to(device)
load_ckpt(G, f'/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/Dis_vector_model/emb_models/Dis_Vector.ckpt')
config.dim_pit = config.dim_con+config.dim_pit

F = F0_Converter(config).eval().to(device)
load_ckpt(F, f'/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/Dis_vector_model/emb_models/pitch_converter.ckpt')


def convert_sp(model, rhy_input, con_input, pit_input, tim_input):
    rhy_code ,emb_r= model.rhythm(rhy_input)
    con_code,emb_c,pit_code,emb_p= model.content_pitch(torch.cat((con_input, pit_input), dim=-1), rr=False)
    sp_output = model.decode(con_code, rhy_code, pit_code, tim_input, T)
    
    return sp_output,emb_r,emb_c,emb_p

def convert_pit(model, rhy_input, con_input, pit_input):
    pit_input = torch.cat([con_input, pit_input], dim=-1)
    rhy_input = torch.nn.functional.pad(rhy_input, (0, 0, 0, T-rhy_input.size(1), 0, 0))
    pit_input = torch.nn.functional.pad(pit_input, (0, 0, 0, T-pit_input.size(1), 0, 0))
    pit_input = model(rhy_input, pit_input, rr=False) # disable random resampling at inference time

    return pit_input

conds = ['R', 'F', 'U']

def extract_embedding_from_wav(wav_file):
    model_G = G  #global
    model_F = F  #global
    conds = ['R', 'F', 'U']   #local
    src_wav, _ = read(wav_file)
    
    with torch.no_grad():
        src_rhy, src_con, src_pit, src_tim = create_feats(src_wav, conds, 0, config)
        inp_rhy, inp_con, inp_pit, inp_tim = src_rhy, src_con, src_pit, src_tim
        
        if 'R' not in conds :
            inp_rhy = src_rhy
        
        if 'U' not in conds :
            inp_tim = src_tim
        
        if 'F' not in conds :
            inp_pit = convert_pit(model_F, src_rhy, src_con, src_pit) 
        
        out_sp, emb_r, emb_c, emb_p = convert_sp(model_G, inp_rhy, inp_con, inp_pit, inp_tim)
        
        # Flatten each embedding tensor
        emb_r_flat = emb_r.view(-1)
        emb_c_flat = emb_c.view(-1)
        emb_p_flat = emb_p.view(-1)
        
        # Concatenate the flattened embeddings into a 1D tensor
        embedding = torch.cat([emb_r_flat, emb_c_flat, emb_p_flat]).unsqueeze(0) 
        
        # Pad the tensor with zeros to reach the desired length
        padding_length = 512 - embedding.shape[1]
        padding = torch.zeros((1, padding_length))
        embedding = torch.cat([embedding, padding], dim=1)
        
        return embedding


# embedding = extract_embedding_from_wav('data/test/p225_001.wav', G, F, config, conds)
# print(embedding.shape)
# print(embedding)