import cv2
import onnx
import onnx_tensorrt.backend as backend

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPE = trt.float16

import numpy as np

from transforms import get_affine_transform, xywh2cs
from inference import get_final_preds

def allocate_buffers(engine):
    print('allocate buffers')
    
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    return h_input, d_input, h_output, d_output

def do_inference(context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)

    # Run inference.
    # st = time.time()
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
    # print('Inference time {} [msec]'.format((time.time() - st)*1000))

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)
    
    return h_output

start = cv2.getTickCount()

with open('./checkpoint/res50_256_fp16.trt', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    h_input, d_input, h_output, d_output = allocate_buffers(engine)

time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('[INFO] model loading time: %.3fms'%time)

img_orig = cv2.imread('./data/pose_test_00228.png', 1)
x,y,w,h = (40,101,729,648) #(326,194,101,330)
c, s = xywh2cs(x,y,w,h)

def standardization(np_input, 
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    np_mean = np.array(mean, dtype=np.float16).reshape(3,1,1)
    np_std = np.array(std, dtype=np.float16).reshape(3,1,1)
    return (np_input - np_mean) / np_std

proc_time_sum = 0
for i in range(50):
    start = cv2.getTickCount()

    trans = get_affine_transform(c, s, 0, (192,256), inv=0)
    warp_img = cv2.warpAffine(img_orig,trans,(192,256),flags=cv2.INTER_LINEAR)
    np_input = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
    np_input = np.expand_dims(np_input, 0).astype(np.float16)
    
    np_inputs_nchw = np_input.transpose(0,3,1,2) / 255
    np_inputs = np.zeros(shape=(1, 3, 256, 192), dtype=np.float16)
    # np_inputs = np_inputs_nchw
    np_inputs[0] = standardization(np_inputs_nchw[0])
    # print (np_inputs_nchw.shape, np_inputs_nchw.dtype)
    np.copyto(h_input, np_inputs.astype(trt.nptype(DTYPE)).ravel())
    # load_input(args.img, h_input)
    
    with engine.create_execution_context() as context:
        output_data = do_inference(context, h_input, d_input, h_output, d_output)
    # output_data = engine.run(np_inputs)[0]
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    proc_time_sum += time

# print (output_data.shape)
print ('[INFO] processing time:%.3fms'%(proc_time_sum/50))

# exit()
output_data = output_data.reshape((1,17,64,48))
preds, maxvals = get_final_preds(output_data, [c], [s])

keypoints = []
cnt_num_point = 0
for idx, ptval in enumerate(zip(preds[0], maxvals[0])):
    point, maxval = ptval
    x,y = point
    # print (x,y, maxval)
    if maxval > 0.5:
        keypoints.extend([x,y,2])
        cnt_num_point += 1
        cv2.circle(img_orig, (x,y), 4, (0,0,255), -1)
    else:
        keypoints.extend([0,0,0])

x,y,w,h = (40,101,729,648) #(326,194,101,330)
cv2.rectangle(img_orig, (x,y), (x+w,y+h), (0,255,0), 1)

cv2.imshow('show', img_orig)
# cv2.imshow('mask', (output_data.squeeze()*255).astype(np.uint8))
cv2.waitKey()

# # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
# h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)
# h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)
# # Allocate device memory for inputs and outputs.
# d_input = cuda.mem_alloc(h_input.nbytes)
# d_output = cuda.mem_alloc(h_output.nbytes)
# # Create a stream in which to copy inputs/outputs and run inference.
# stream = cuda.Stream()

# with engine.create_execution_context() as context:
#     # Transfer input data to the GPU.
#     cuda.memcpy_htod_async(d_input, h_input, stream)
#     # Run inference.
#     context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
#     # Transfer predictions back from the GPU.
#     cuda.memcpy_dtoh_async(h_output, d_output, stream)
#     # Synchronize the stream
#     stream.synchronize()
#     # Return the host output. 
# # return h_output
