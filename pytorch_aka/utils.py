import torch
from torch import nn
import torch.nn.functional as F

def Input(param):
    return None

def ReLU(param):
    slope = float(param)
    if slope == 0:
        return nn.ReLU()
    else:
        return nn.LeakyReLU(slope)

def Split(param):
    n = int(param)
    def call(x):
        return [x for i in range(n)]
    return call

def Softmax(param):
    axis = int(param)
    if axis >= 0:
        axis += 1
    def call(x):
        return F.softmax(x,dim=axis)
    return call
    return nn.Softmax(dim=axis)

def Dropout(param):
    scale = float(param)
    def call(x):
        return scale*x
    return call

def Concat(param):
    axis = int(param)
    if axis >= 0:
        axis += 1
    def call(*x):
        return torch.cat(x,dim=axis)
    return call

def Pooling(param):
    param = [int(x) for x in param.split(' ')[:-1]]
    pooling_type,kernel_w,kernel_h,stride_w,stride_h,pad_left,pad_right,pad_top,pad_bottom,global_pooling,pad_mode,avgpool_count_include_pad,adaptive_pooling,out_w,out_h = param
    # pad mode的问题还没解决，看情况
    if global_pooling:
        if pooling_type == 0:
            return nn.AdaptiveMaxPool2d(1)
        elif pooling_type == 1:
            return nn.AdaptiveAvgPool2d(1)

    if adaptive_pooling:
        if pooling_type == 0:
            return nn.AdaptiveMaxPool2d((out_h,out_w))
        elif pooling_type == 1:
            return nn.AdaptiveAvgPool2d((out_h,out_w))

    if pooling_type == 0:
        return nn.MaxPool2d(kernel_size=(kernel_h,kernel_w),stride=(stride_h,stride_w),padding=(pad_top,pad_left))
    elif pooling_type == 1:
        return nn.AvgPool2d(kernel_size=(kernel_h,kernel_w),stride=(stride_h,stride_w),padding=(pad_top,pad_left),count_include_pad=avgpool_count_include_pad)

    return None

def Convolution(param):
    param = [int(x) for x in param.split(' ')[:-1]]
    num_output,kernel_w,kernel_h,dilation_w,dilation_h,stride_w,stride_h,pad_left,pad_right,pad_top,pad_bottom,pad_value,bias_term,weight_data_size,activation_type = param[:15]
    
    block = []
    block.append(nn.Conv2d(weight_data_size//num_output//kernel_w//kernel_h, num_output, (kernel_h,kernel_w), stride=(stride_h,stride_w), padding=(pad_top,pad_left), dilation=(dilation_h,dilation_w), bias=bias_term))
    if activation_type == 0:
        pass
    elif activation_type == 1:
        block.append(nn.ReLU())
    elif activation_type == 2:
        block.append(nn.LeakyReLU(param[15]))
    elif activation_type == 4:
        block.append(nn.Sigmoid())
    else:
        print('fuck Convolution')
    block = nn.Sequential(*block)

    return block