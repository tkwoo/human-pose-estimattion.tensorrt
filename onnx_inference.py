import cv2
import onnx
# import onnx_tensorrt.backend as backend
import onnxruntime as rt
from onnxruntime.datasets import get_example
import onnxruntime.backend as backend
from onnx import load
import numpy as np
from transforms import get_affine_transform, xywh2cs
from inference import get_final_preds


start = cv2.getTickCount()
# model = onnx.load("./checkpoint/res50_256.onnx")
# name = get_example('./checkpoint/res50_256.onnx')
model = load('./checkpoint/res50_256.onnx')

rep = backend.prepare(model)
# sess = rt.InferenceSession(model)
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name
# engine = backend.prepare(model, device='CUDA:1')
time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('[INFO] model loading time: %.3fms'%time)


img_orig = cv2.imread('./data/pose_test_00228.png', 1)
x,y,w,h = (40,101,729,648) #(326,194,101,330)
c, s = xywh2cs(x,y,w,h)

def standardization(np_input, 
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    np_mean = np.array(mean, dtype=np.float32).reshape(3,1,1)
    np_std = np.array(std, dtype=np.float32).reshape(3,1,1)
    return (np_input - np_mean) / np_std

proc_time_sum = 0
for i in range(20):
    start = cv2.getTickCount()

    trans = get_affine_transform(c, s, 0, (192,256), inv=0)
    warp_img = cv2.warpAffine(img_orig,trans,(192,256),flags=cv2.INTER_LINEAR)
    np_input = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
    np_input = np.expand_dims(np_input, 0).astype(np.float32)
    
    np_inputs_nchw = np_input.transpose(0,3,1,2) / 255
    np_inputs = np.zeros(shape=(1, 3, 256, 192), dtype=np.float32)
    # np_inputs = np_inputs_nchw
    np_inputs[0] = standardization(np_inputs_nchw[0])
    # print (np_inputs_nchw.shape, np_inputs_nchw.dtype)
    # output_data = engine.run(np_inputs)[0]
    # output_data = sess.run([output_name], {input_name: np_inputs})
    output_data = rep.run(np_inputs)[0]
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    proc_time_sum += time

# print (output_data.shape)
print ('[INFO] processing time:%.3fms'%(proc_time_sum/20))

# exit()
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