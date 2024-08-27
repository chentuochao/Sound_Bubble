import torch
import torch.nn as nn

from .tfgridnet_causal import TFGridNet
import torch.nn.functional as F


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class Net(nn.Module):
    def __init__(self, stft_chunk_size=160, stft_pad_size = 120, stft_back_pad = 0,
                 num_ch=2, D=64, B=6, I=1, J=1, L=0, H=128,
                 use_attn=False, lookahead=True, local_atten_len=100,
                 E = 4, chunk_causal=False, num_src = 1,
                 spectral_masking=False, use_first_ln=False, merge_method = "None",
                 directional = False, conv_lstm = True, lstm_down=5, fb_type='stft'):
        super(Net, self).__init__()
        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.num_ch = num_ch
        self.lookahead = lookahead
        self.stft_back_pad = stft_back_pad

        self.embed_dim = D
        self.E = E

        # Input conv to convert input audio to a latent representation        
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size
        
        nfreqs = self.nfft//2 + 1

        # TF-GridNet        
        self.tfgridnet = TFGridNet(None,
                                   n_srcs=num_src,
                                   n_fft=self.nfft,
                                   look_back = stft_back_pad,
                                   stride=stft_chunk_size,
                                   emb_dim=D,
                                   emb_ks=I,
                                   emb_hs=J,
                                   n_layers=B,
                                   n_imics=num_ch,
                                   attn_n_head=L,
                                   attn_approx_qk_dim=E*nfreqs,
                                   use_attn = use_attn,
                                   lstm_hidden_units=H,
                                   local_atten_len=local_atten_len,
                                   chunk_causal = chunk_causal,
                                   spectral_masking = spectral_masking,
                                   use_first_ln=use_first_ln,
                                   merge_method = merge_method,
                                   directional = directional,
                                   conv_lstm = conv_lstm,
                                   lstm_down=lstm_down,
                                   fb_type=fb_type)

    def init_buffers(self, batch_size, device):
        return self.tfgridnet.init_buffers(batch_size, device)

    def predict(self, x, input_state, pad=True):
        mod = 0
        if pad:
            pad_size = (self.stft_back_pad, self.stft_pad_size) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)

        x, next_state = self.tfgridnet(x, input_state)
        # x = x[..., : -self.stft_pad_size]
        
        if mod != 0:
            x = x[:, :, :-mod]

        return x, next_state

    def forward(self, inputs, input_state = None, pad=True):
        x = inputs['mixture']

        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)

        x, next_state = self.predict(x, input_state, pad)

        return {'output': x, 'next_state': next_state}

if __name__ == "__main__":
    model_params = {
        "stft_chunk_size": 192,
        "stft_pad_size": 96,
        "stft_back_pad": 0,
        "num_ch": 6,
        "D": 16,
        "L": 4,
        "I": 1,
        "J": 1,
        "B": 4,
        "H": 64,
        "E": 2,
        "local_atten_len": 50,
        "use_attn": False,
        "lookahead": True,
        "chunk_causal": True,
        "use_first_ln": True,
        "merge_method": "early_cat",
        "directional": True
    }
    device = torch.device('cpu') ##('cuda')
    model = Net(**model_params).to(device)

    num_chunk = 50
    test_num = 10
    chunk_size = model_params["stft_chunk_size"]
    look_front = model_params["stft_pad_size"]
    look_back = model_params["stft_back_pad"] #model_params["lookback"]
    x = torch.rand(4, 6, look_back + chunk_size*num_chunk + look_front)
    x = x.to(device)
    x2 = x[..., :look_back + chunk_size*test_num + look_front]
    inputs = {"mixture": x}
    inputs2 = {"mixture": x2}
    y = model(inputs, pad=False)['output']
    y2 = model(inputs2, pad=False)['output']

    print(x.shape, y.shape, y2.shape)
    _id  = 3
    check_valid = torch.allclose(y2[:, 0, :chunk_size*test_num], y[:, 0, :chunk_size*test_num], atol=1e-2 )
    print((y2[_id, 0, :chunk_size*test_num] - y[_id, 0, :chunk_size*test_num]).abs().max())
    print(check_valid)
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot( y[_id, 0, :chunk_size*test_num].detach().numpy())
    # plt.plot( y2[_id, 0, :chunk_size*test_num].detach().numpy(), linestyle = '--', color = 'r')
    # plt.show()
