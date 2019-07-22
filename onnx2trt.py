
'''
onnx version
'''
import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("./checkpoint/res50_256.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(1, 3, 256, 192)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)

'''
tensorrt version
'''
# import tensorrt as trt
# import onnx

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# # ONNX_MODEL = './checkpoint/deeplabv3_256.onnx'
# ONNX_MODEL = './checkpoint/unet_wm_256.onnx'

# onnx_model = onnx.load(ONNX_MODEL)
# print (onnx.checker.check_model(onnx_model))
# print (onnx.helper.printable_graph(onnx_model.graph))

# def build_engine():
#     with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#         # Configure the builder here.
#         builder.max_workspace_size = 2**30
#         # Parse the model to create a network.
#         with open(ONNX_MODEL, 'rb') as model:
#             parser.parse(model.read())
#             print (parser.get_error(index=0))
#         # Build and return the engine. Note that the builder, network and parser are destroyed when this function returns.
#         return builder.build_cuda_engine(network)

# engine = build_engine()