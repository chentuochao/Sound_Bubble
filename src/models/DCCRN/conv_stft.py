


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
import torchaudio


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)#**0.5
    
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        #inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
        # print(self.weight.shape)
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)
        # print(outputs.shape)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim//2+1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2+imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 

        if phase is not None:
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        #print(self.weight.shape, inputs.shape)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride, padding = 0) 

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1,1,inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride, padding = 0)
        outputs = outputs/(coff+1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        # outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        
        return outputs


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


def test_casual():
    torch.manual_seed(20)
    win_len = 512
    win_inc = 100 
    fft_len = 512
    feat_num = win_len//2 + 1
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hann', feature_type='complex')
    ifft = ConviSTFT(win_len, win_inc, fft_len=fft_len, win_type='hann', feature_type='complex')


    CHUNK = win_inc*5
    inputs = torch.rand(12, 1, CHUNK*4)
    ### input all data one time
    begin_idx = win_len - win_inc - win_inc
    pad_size = (win_len - win_inc - win_inc, win_inc) 
    inputs1 = F.pad(inputs, pad_size)
    print(inputs.shape)
    outputs1 = fft(inputs1)
    print(outputs1.shape)

    ### input data chunk by chunk
    for i in range(0, 3):
        come_data = inputs1[:, :, i*CHUNK: begin_idx + (i+1)*CHUNK+ win_inc]
        y = fft(come_data)
        ref_y = outputs1[:, :, i*5: (i+1)*5]
        
        check_valid = torch.allclose(y, ref_y, rtol=1e-2)
        print(check_valid)

def test_fft():
    torch.manual_seed(20)
    win_len = 512
    win_inc = 64 
    fft_len = 512
    feat_num = win_len//2 + 1
    
    inputs = torch.rand(1,1,128)
    #inputs, sample_rate = torchaudio.load("test.wav")
    #inputs = torchaudio.functional.resample(inputs, sample_rate, sample_rate//2)
    #inputs = inputs[..., :200]
    #inputs = inputs.unsqueeze(0)
    pad_size = (win_len - win_inc - win_inc, win_inc) 
    inputs = F.pad(inputs, pad_size)
    # inputs, mod = mod_pad(inputs, chunk_size=win_inc, pad=pad_size)


    print(inputs.shape, (inputs.shape[-1] - fft_len)//win_inc + 1 )
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hann', feature_type='complex')
    ifft = ConviSTFT(win_len, win_inc, fft_len=fft_len, win_type='hann', feature_type='complex')

    import librosa

    outputs0 = fft(inputs)
    np_inputs = inputs.numpy().reshape([-1])
    librosa_stft = librosa.stft(np_inputs, win_length=win_len, n_fft=fft_len, hop_length=win_inc, center=False)
    outputs1 = outputs0.numpy()[0]

    print(outputs1.shape, librosa_stft.shape)
    out_pad = int(np.ceil(win_len/win_inc)) - 1
    print(out_pad)

    print( "Real: ", np.allclose(np.real(librosa_stft), outputs1[:feat_num] , atol=1e-03) )    
    print( "Img: ", np.allclose(np.imag(librosa_stft), outputs1[feat_num:]  , atol=1e-03) )    
     
    #outputs0 =  F.pad(outputs0, (out_pad, 0)) 
    print(outputs0.shape)
    input_reverse = ifft(outputs0)
    #input_reverse = input_reverse[:, :, win_len - 2*win_inc: -win_inc]
    print(input_reverse.shape)
    print( "Real: ", np.allclose(inputs, input_reverse , atol=1e-03) )    

def test_ifft1():
    import soundfile as sf
    N = 400
    inc = 100
    fft_len=512
    torch.manual_seed(N)
    data = np.random.randn(16000*8)[None,None,:]
#    data = sf.read('../ori.wav')[0]
    inputs = data.reshape([1,1,-1])
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs1 = fft(inputs)
    print(outputs1.shape) 
    outputs2 = ifft(outputs1)
    sf.write('conv_stft.wav', outputs2.numpy()[0,0,:],16000)
    print('wav MSE', torch.mean(torch.abs(inputs[...,:outputs2.size(2)]-outputs2)**2))


def test_ifft2():
    N = 400
    inc = 100
    fft_len=512
    np.random.seed(20)
    torch.manual_seed(20)
    t = np.random.randn(16000*4)*0.001
    t = np.clip(t, -1, 1)
    #input = torch.randn([1,16000*4]) 
    input = torch.from_numpy(t[None,None,:].astype(np.float32))
    
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    
    out1 = fft(input)
    output = ifft(out1)
    print('random MSE', torch.mean(torch.abs(input-output)**2))
    import soundfile as sf
    sf.write('zero.wav', output[0,0].numpy(),16000)


if __name__ == '__main__':
    test_casual()
    # test_ifft1()
    #test_ifft2()
