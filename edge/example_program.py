import sounddevice as sd
import queue
import numpy as np
import os, glob
import json
import torch
import threading
import onnxruntime as ort
from edge.flatbuf import flatten_state_buffers
from scipy.io.wavfile import write as wavwrite
from scipy.signal import resample
import time
import src.utils as utils

def load_net(path) -> torch.nn.Module:
    with open(path, 'rb') as f:
        baseline_cfg = json.load(f)

    device = 'cpu'
    model_params = baseline_cfg['pl_module_args']['model_params']

    mdl = utils.import_attr(baseline_cfg['pl_module_args']['model'])(**model_params)
    mdl = mdl.to(device)

    return mdl, model_params


def write_audio_file(file_path, data, sr):
    """
    Writes audio file to system memory.
    @param file_path: Path of the file to write to
    @param data: Audio signal to write (n_channels x n_samples)
    @param sr: Sampling rate
    """
    wavwrite(file_path, sr, data.T)

# STREAM PARAMS
_model, params = load_net('configs/TFG_rt.json')
CHUNK_SIZE = params['stft_chunk_size']
PAD_SIZE = params['stft_pad_size']
NUM_CHANNELS = params['num_ch']

sr = 48000 # Sampling rate of input audio
tgt_sr = 24000 # Sampling rate of processing audio

downsample_rate = sr // tgt_sr

BLOCKSIZE = int(round(8e-3 * sr)) # 8 ms @ 48kHz

activated = False
running = 0
overrun = 0


in_queue = queue.Queue()
ZEROS = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
OUT_BUF = None

state_buffers = _model.init_buffers(1, 'cpu')
tse_path = 'models/ONNX/model.onnx'

class ModelWrapper():
    def __init__(self, tse_path, state_buffers):
        self.buffer_names, self.buffers = flatten_state_buffers(state_buffers)
        
        self.current_inputs = dict()
        for buf_name, buf in zip(self.buffer_names, self.buffers):
            self.current_inputs[buf_name] = buf.detach().numpy()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2

        self.tse = ort.InferenceSession(tse_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

    def infer(self, x: np.ndarray) -> np.ndarray:
        
        self.current_inputs['mixture'] = x
        
        t1 = time.time()
        outputs = self.tse.run(None, self.current_inputs)
        t2 = time.time()
        if t2 - t1 > 8e-3:
            print('TOO LONG', t2 - t1)
        
        for i in range(1, len(self.buffer_names)+1):
            self.current_inputs[self.buffer_names[i-1]] = outputs[i]

        return outputs[0]

def stream_callback(indata, outdata, frames, time, status):
    global OUT_BUF
    if OUT_BUF is not None:
        outdata[:] = OUT_BUF.T
        OUT_BUF = None
    else:
        outdata[:] = ZEROS
    
    in_queue.put(indata.T.copy())

# Initialize stream
stream = sd.Stream(samplerate=sr,
                   latency='low',
                   channels=(8, 2),
                   blocksize=BLOCKSIZE,
                   callback=stream_callback,
                   dtype=np.float32)

model = ModelWrapper(tse_path, state_buffers)

def stop():
    global running
    running = False
    stream.stop()
    stream.close()

def cmd_input(message):
    cmdlist = input(message).strip().split(' ')
    cmdlist = [x for x in cmdlist if x != '']
    
    if len(cmdlist) == 0:
        return cmd_input(message)
    
    if cmdlist[0] == 'cancel':
        print("Cancelled")
        return None
    
    return cmdlist

started = threading.Event()

def respond_to_inputs():
    global enrolling
    while True:
        cmdlist = cmd_input("Enter command [start|stop|activate]:")

        cmd = cmdlist[0]
        if cmd == 'start':
            started.set()
        elif cmd == 'stop':
            stop()
            break
        else:
            print(f"Command {cmd} not recognized")

def calibrate_channels(server):
    print(f"CALIBRATING SERVER {[server['name']]}")
    print(f"Tap all microphones in order from RIGHT to LEFT")

    order = []
    client = Client(server['host'], server['port'])

    for i in range(6):
        client.start_streaming()
        print(f"TAP MIC {i+1} FROM RIGHT")

        done = False
        while not done:
            chunk = client.mic_stream.get()
            amp = np.abs(chunk) / (32767)
            channelwise_max_amp = amp.max(axis=1) 
            if (channelwise_max_amp >= 0.8).any():
                channel = np.argmax(channelwise_max_amp)
                if channel not in order:
                    order.append(int(channel))
                    done = True       

        client.stop_streaming()
    
    return order

while True:
    threading.Thread(target=respond_to_inputs).start()

    started.wait()
    print("STARTED")
    
    running = True
    
    odata = []
    
    T = CHUNK_SIZE
    
    current_frame = np.zeros((1, 2, T + PAD_SIZE), dtype=np.float32)
    data = []
    odata = []

    stream.start()

    # Calibrate
    calibration_finished = np.zeros(8)
    calibration_order = []
    while len(calibration_order) < NUM_CHANNELS:
        input_data = in_queue.get(timeout=0.015)
        
        channel = None
        for ch in range(calibration_finished.shape[0]):
            if calibration_finished[ch]:
                continue

            if np.abs(input_data[ch]).max() > 0.8:
                channel = ch
                break
        
        calibration_finished[channel] = True
        calibration_order.append(channel)

    print("Calibration finished, order is", calibration_order)

    while running:
        try:
            input_data = in_queue.get(timeout=0.015)
            input_data = input_data[calibration_order]

            assert input_data.shape[0] == NUM_CHANNELS

            if activated:
                # Resample to 24kHz
                input_data_resampled = resample(input_data, CHUNK_SIZE, axis=-1)
                # input_data_resampled = input_data
                
                current_frame = np.roll(current_frame, shift=-T, axis=-1)
                current_frame[0, :, -T:] = input_data_resampled

                y = model.infer(current_frame)
                #y = input_data_resampled
            else:                
                y = input_data
            
            OUT_BUF = y
            
            data.append(input_data)
            odata.append(y)
        except queue.Empty:
            pass
        
    data = np.concatenate(data, axis=1)
    odata = np.concatenate(odata, axis=1)
    # print(data.shape)
        
    write_audio_file('saved_input.wav', data, sr)
    write_audio_file('saved_output.wav', odata, sr)


    
