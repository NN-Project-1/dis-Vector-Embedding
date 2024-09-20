import torch
import torch.nn as nn
import torch.nn.functional as F
     
class InterpLnr(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.max_len_seq = config.max_len_seq
        self.max_len_pad = config.max_len_pad
        
        self.min_len_seg = config.min_len_seg
        self.max_len_seg = config.max_len_seg
        
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1
        self.training = config.train
        
        
    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]
            
        return out_tensor 
    

    def forward(self, x, len_seq):  
        
        if not self.training:
            return x
        
        device = x.device
        batch_size = x.size(0)
        indices = torch.arange(self.max_len_seg*2, device=device)\
                  .unsqueeze(0).expand(batch_size*self.max_num_seg, -1)
        scales = torch.rand(batch_size*self.max_num_seg, 
                            device=device) + 0.5
        
        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        
        len_seg = torch.randint(low=self.min_len_seg, 
                                high=self.max_len_seg, 
                                size=(batch_size*self.max_num_seg,1),
                                device=device)
        idx_mask = idx_scaled_fl < (len_seg - 1)
       
        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)

        offset = F.pad(offset[:, :-1], (1,0), value=0).view(-1, 1)
        
        idx_scaled_org = idx_scaled_fl + offset
        
        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)
        
        idx_mask_final = idx_mask & idx_mask_org
        
        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        
        index_1 = torch.repeat_interleave(torch.arange(batch_size, 
                                            device=device), counts)
        
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1
        
        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        
        y = (1-lambda_f)*y_fl + lambda_f*y_cl
        
        sequences = torch.split(y, counts.tolist(), dim=0)
       
        seq_padded = self.pad_sequences(sequences)
        
        return seq_padded    


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


    
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
    
    
class Encoder_R(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_neck_2 = config.dim_neck_2
        self.freq_2 = config.freq_2
        self.dim_rhy = config.dim_rhy
        self.dim_enc_2 = config.dim_enc_2
        self.dim_emb = config.dim_spk_emb
        self.chs_grp = config.chs_grp
        self.dropout = config.dropout
        
        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_rhy if i == 0 else self.dim_enc_2,
                         self.dim_enc_2,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_2 // self.chs_grp, self.dim_enc_2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True)
        self.embedding_dim = 64
        self.embedding_layer = None 

    def forward(self, x, mask):
                
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2:]
            
        codes = torch.cat((out_forward[:, self.freq_2 - 1::self.freq_2, :], out_backward[:, ::self.freq_2, :]), dim=-1)
        flattened_codes = codes.flatten(start_dim=1)

        if self.embedding_layer is None:
            expected_flattened_dim = flattened_codes.shape[1]
           
            self.embedding_layer = nn.Linear(expected_flattened_dim, self.embedding_dim).to(flattened_codes.device)
        emb = self.embedding_layer(flattened_codes)
        return codes, emb
    

class Encoder_CN_PI(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_neck = config.dim_neck
        self.freq = config.freq
        self.freq_3 = config.freq_3
        self.dim_enc = config.dim_enc
        self.dim_enc_3 = config.dim_enc_3
        self.dim_con = config.dim_con
        self.dim_pit = config.dim_pit
        self.chs_grp = config.chs_grp
        self.register_buffer('len_org', torch.tensor(config.max_len_pad))
        self.dim_neck_3 = config.dim_neck_3
        self.dim_f0 = config.dim_f0

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_con if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
        self.convolutions_1 = nn.ModuleList(convolutions)

        self.lstm_1 = nn.LSTM(self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True)

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_pit if i == 0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3))
            convolutions.append(conv_layer)
        self.convolutions_2 = nn.ModuleList(convolutions)

        self.lstm_2 = nn.LSTM(self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True)

        self.interp = InterpLnr(config)
        self.embedding_layer = None 
        self.embedding_dim_x=256
        self.embedding_dim_f0=128

    def forward(self, x_f0, rr=True):

        x = x_f0[:, :self.dim_con, :]
        f0 = x_f0[:, self.dim_con:, :]

        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            x = F.relu(conv_1(x))
            f0 = F.relu(conv_2(f0))
            x_f0 = torch.cat((x, f0), dim=1).transpose(1, 2)
            if rr:
                x_f0 = self.interp(x_f0, self.len_org.expand(x.size(0)))
            x_f0 = x_f0.transpose(1, 2)
            x = x_f0[:, :self.dim_enc, :]
            f0 = x_f0[:, self.dim_enc:, :]

        x_f0 = x_f0.transpose(1, 2)
        x = x_f0[:, :, :self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc:]

        self.lstm_1.flatten_parameters()
        self.lstm_2.flatten_parameters()
        x = self.lstm_1(x)[0]
        f0 = self.lstm_2(f0)[0]

        x_forward = x[:, :, :self.dim_neck]
        x_backward = x[:, :, self.dim_neck:]

        f0_forward = f0[:, :, :self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3:]

        codes_x = torch.cat((x_forward[:, self.freq - 1::self.freq, :],
                             x_backward[:, ::self.freq, :]), dim=-1)

        codes_f0 = torch.cat((f0_forward[:, self.freq_3 - 1::self.freq_3, :],
                              f0_backward[:, ::self.freq_3, :]), dim=-1)
        flattened_codes_x = codes_x.flatten(start_dim=1)
 
        expected_flattened_dim = flattened_codes_x.shape[1]
        self.embedding_layer = nn.Linear(expected_flattened_dim, self.embedding_dim_x).to(flattened_codes_x.device)
        emb_1 = self.embedding_layer(flattened_codes_x)
        flattened_codes_f0 = codes_f0.flatten(start_dim=1)
        self.embedding_layer = nn.Linear(expected_flattened_dim, self.embedding_dim_f0).to(flattened_codes_x.device)
        emb_2 = self.embedding_layer(flattened_codes_f0)

        return codes_x, emb_1, codes_f0, emb_2

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_neck = config.dim_neck 
        self.dim_neck_2 = config.dim_neck_2 
        self.dim_emb = config.dim_spk_emb
        self.dim_freq = config.dim_freq
        self.dim_neck_3 = config.dim_neck_3 
        self.lstm = nn.LSTM(256,512, 3, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(1024, self.dim_freq)

    def forward(self, x):
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)
        return decoder_output          

class Encoder_TI(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_neck_2 = config.dim_neck_2  
        self.freq_2 = config.freq_2 
        self.dim_tim = 37  
        self.dim_enc_2 = config.dim_enc_2  
        self.dim_emb = config.dim_spk_emb 
        self.chs_grp = config.chs_grp  
        self.dropout = config.dropout  
        
        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_tim if i == 0 else self.dim_enc_2,
                         self.dim_enc_2,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_2 // self.chs_grp, self.dim_enc_2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True)
        self.embedding_dim = 64
        self.embedding_layer = None  

    def forward(self, x, mask):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2:]
            
        codes = torch.cat((out_forward[:, self.freq_2 - 1::self.freq_2, :], out_backward[:, ::self.freq_2, :]), dim=-1)
        flattened_codes = codes.flatten(start_dim=1)
        if self.embedding_layer is None:
            expected_flattened_dim = flattened_codes.shape[1]
      
            self.embedding_layer = nn.Linear(expected_flattened_dim, self.embedding_dim).to(flattened_codes.device)
        emb_3 = self.embedding_layer(flattened_codes)

        return codes, emb_3   
    

class Vector_Mode(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.encoder_1 = Encoder_CN_PI(config)
        self.encoder_2 = Encoder_TI(config)
        self.encoder_3 = Encoder_R(config)
        self.decoder = Decoder(config)
    
        self.freq = config.freq
        self.freq_2 = config.freq_2
        self.freq_3 = config.freq_3


    def forward(self, x_f0, x_org, c_trg,tim_f, rr=True):

        x_1 = x_f0.transpose(2,1)
        codes_x, emb_1,codes_f0 ,emb_2= self.encoder_1(x_1, rr)
        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        tim_f=tim_f.transpose(2,1)
        tim_code,emb_3 =self.encoder_3(tim_f,None)
        x_2 = x_org.transpose(2,1)
        codes_2 ,emb_4 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        encoder_outputs = torch.cat((code_exp_1, code_exp_2, code_exp_3,tim_code),dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs
    
    def rhythm(self, x_org):
        x_2 = x_org.transpose(2,1)
        codes_2 ,emb_4= self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        
        return code_exp_2 ,emb_4

    def timbre(self,tim_f):
        tim_f=tim_f.transpose(2,1)
        tim_code,emb_3 =self.encoder_3(tim_f,None)
        return tim_code,emb_3

    def content_pitch(self, x_f0, rr=True):
        x_1 = x_f0.transpose(2,1)
        codes_x, emb_1, codes_f0 ,emb_2 = self.encoder_1(x_1, rr)
        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        
        return code_exp_1, emb_1, code_exp_3,emb_2

    def decode(self, code_exp_1, code_exp_2, code_exp_3, tim_f):
        encoder_outputs = torch.cat((code_exp_1, code_exp_2, code_exp_3,tim_f),dim=-1)
        print("inference encoder_outputs :",encoder_outputs.shape)
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs


  