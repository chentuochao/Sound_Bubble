import torch
import random
import numpy as np
from src.utils import load_net


# Model Wrapper
class ModelWrapper(torch.nn.Module):
    def __init__(self, mdl) -> None:
        super().__init__()

        self.model = mdl
        self.internal_state = None

    def feed(self, mix, pad=False) -> torch.Tensor:
        if self.internal_state is None:
            self.internal_state = self.model.init_buffers(mix.shape[0], device='cpu')
        
        outputs = self.model(dict(mixture=mix), self.internal_state, pad=pad)

        out = outputs['output']
        self.internal_state = outputs['next_state']

        # print(self.internal_state['conv_buf'])

        return out

def streaming_inference(mdl: ModelWrapper, X: torch.Tensor, chunk_size: int, pad_length: int):
    num_samples = X.shape[-1]
    T = chunk_size

    current_frame = torch.zeros((1, X.shape[1], T + pad_length))
    current_frame[..., -pad_length:] = X[..., :pad_length]
    
    outputs = []
    k = 0
    for i in range(pad_length, num_samples - pad_length + 1, T):
        k +=1
        current_frame = torch.roll(current_frame, shifts=-T, dims=-1)
        current_frame[..., -T:] = X[..., i:i+T]
        out_chunk = mdl.feed(current_frame)
        # print(Y[..., :5])
        outputs.append(out_chunk)

    output = torch.cat(outputs, dim=-1)
    
    return output

if __name__ == "__main__":

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    mdl, params = load_net('experiments/TFG_rw_IPD.json', return_params=True)
    model_params = params['pl_module_args']['model_params']
    mdl = mdl.model

    CHUNK_SIZE = model_params['stft_chunk_size']
    PAD_SIZE = model_params['stft_pad_size']
    num_chunks = 200

    T = CHUNK_SIZE
    C = model_params['num_ch']
    B = 1

    mdl_os = ModelWrapper(mdl)
    mdl_os.eval()

    mdl_stream = ModelWrapper(mdl)
    mdl_stream.eval()

    X = torch.randn(B, C, T * num_chunks + PAD_SIZE) * 10

    with torch.no_grad():
        Y = mdl_os.feed(X, pad=False)
        
        # Z = mdl_stream(X)
        Z = streaming_inference(mdl_stream, X, chunk_size=T, pad_length=PAD_SIZE)

        print(Z.shape, Y.shape)

    print(Y[..., 176*CHUNK_SIZE:176*CHUNK_SIZE+5])
    print(Z[..., 176*CHUNK_SIZE:176*CHUNK_SIZE+5])
    print("Test successful:", torch.allclose(Y, Z, atol=1e-3))
    print("Max diff:", torch.max(torch.abs(Z - Y)) )
