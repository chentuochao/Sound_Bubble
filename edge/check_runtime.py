import torch
import onnxruntime as ort
import numpy as np
import time
#import pyarmnn as ann
from edge.edge_utils import load_inputs

import tflite_runtime.interpreter as tflite


RUNS = 10

def eval_torch(model_file, mixture: np.ndarray, buffer_names: list, buffers: np.ndarray):
    model = torch.jit.load(model_file)

    X = torch.from_numpy(mixture)
    
    bufs = []
    for i in range(len(buffer_names)):
        bufs.append(torch.from_numpy(buffers[i]))

    model.eval()
    with torch.no_grad():
        t1 = time.time()
        
        for i in range(RUNS):
            Y = model(X, *bufs)[0]
        
        t2 = time.time()

    return Y.numpy(), (t2 - t1) / RUNS

def eval_onnx(model_file, mixture: np.ndarray, buffer_names: list, buffers: np.ndarray, dtype = np.float32):
    sess_opt = ort.SessionOptions()
    #sess_opt.intra_op_num_threads=1
    # sess_opt.enable_profiling = True
    ort_sess = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'], sess_options=sess_opt)

    inputs = {}

    inputs['mixture'] = mixture.astype(dtype)
    for name, buffer in zip(buffer_names, buffers):
        inputs[name] = buffer.astype(dtype)
    
    t1 = time.time()
    
    for i in range(RUNS):
        Y = ort_sess.run(None, inputs)[0]

    t2 = time.time()
    
    return Y, (t2 - t1) / RUNS


def eval_tflite(model_file, mixture: np.ndarray, buffer_names: list, buffers: np.ndarray, dtype = np.float32, use_armnn=False):
    if use_armnn:
        arm_delegate = tflite.load_delegate(library="armnn/libarmnnDelegate.so",
                                            options={"backends": "CpuAcc,CpuRef", "logging-severity":"info"})

        interpreter = tflite.Interpreter(model_path=model_file, experimental_delegates=[arm_delegate])
    else:
        interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    mixture = np.transpose(mixture.astype(dtype), axes=(0,2,1))

    input_dict = {x['name']:x for x in interpreter.get_input_details()}
    
    for name, buf in zip(buffer_names, buffers):
        name = name.replace(':', '__')
        interpreter.set_tensor(input_dict[name]['index'], buf)

    interpreter.set_tensor(input_dict['mixture']['index'], mixture)

    output_dict = {x['name']:x for x in interpreter.get_output_details()}
    filtered_output = output_dict['filtered_output']

    t1 = time.time()
    
    for i in range(RUNS):
        interpreter.set_tensor(input_dict['mixture']['index'], mixture)

        interpreter.invoke()

        Y = interpreter.get_tensor(filtered_output['index'])

    t2 = time.time()
    
    return Y, (t2 - t1)/RUNS

# def eval_armnn(model_file, mixture: np.ndarray, buffer_names: list, buffers: np.ndarray, dtype = np.float32, use_armnn=False):
#     # Load network
#     parser = ann.ITfLiteParser()
#     network = parser.CreateNetworkFromBinaryFile(model_file)

#     # Create runtime
#     options = ann.CreationOptions()
#     runtime = ann.IRuntime(options)

#     # Backend choices earlier in the list have higher preference.
#     preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
#     opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

#     # Load the optimized network into the runtime.
#     net_id, _ = runtime.LoadNetwork(opt_network)
#     print("Loaded network, id={net_id}")
    
#     # Create an inputTensor for inference.
#     graph_id = 0
#     input_names = parser.GetSubgraphInputTensorNames(graph_id)
#     input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
#     input_tensor_id = input_binding_info[0]
#     input_tensor_info = input_binding_info[1]
#     print('tensor id: ' + str(input_tensor_id))
#     print('tensor info: ' + str(input_tensor_info))

    
#     return np.zeros_like(mixture), 0


mixture, buffer_names, buffers = load_inputs('models/test_data/replication_test')
print(mixture.shape)

# print("[ArmNN]")
# y_armnn, t_armnn = eval_armnn('models/tf/tse_float32.tflite', mixture, buffer_names, buffers)

print("[TFLITE]")
y_tflite, t_tflite = eval_tflite('models/tf/model_float32.tflite', mixture, buffer_names, buffers)

print("[TORCH]")
y_torch, t_torch = eval_torch('models/TorchJIT/model.pt', mixture, buffer_names, buffers)

print("[ONNX]")
y_onnx, t_onnx = eval_onnx('models/ONNX/model.onnx', mixture, buffer_names, buffers)

#print("[ONNX (int8)]")
#y_onnx_simp, t_onnx_simp = eval_onnx('models/model_quant.onnx', X, E, enc_buf, dec_buf, out_buf)

#print("[MNN]")
#y_mnn, t_mnn = eval_mnn('model.mnn', X, E, enc_buf, dec_buf, out_buf)

print(f"Torch: {t_torch * 1000}ms")
print(f"ONNX: {t_onnx * 1000}ms")
print(f"TFLite: {t_tflite * 1000}ms")
# print(f"ArmNN: {t_armnn * 1000}ms")
#print(f"MNN: {t_mnn * 1000}ms")


print('Torch', y_torch[0, 0, :10])
print('ONNX', y_onnx[0, 0, :10])
print('TFLite', y_tflite[0, :10, 0])
# print(f"ArmNN", y_armnn[0, :10, 0])
#print('MNN', y_mnn[0, 0, :10])


