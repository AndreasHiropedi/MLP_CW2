from model_architectures import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Test whether all the output of the Batch Normalisation layer has mean 0 and variance 1 
# And test whether Residual Connection is correctly implemented in the Processing block

# Initialize flag as True

all_passed = True

# Initialize the test block
input_shape = (100,3,32,32)
filters = 3
kernel_size = 3
padding, bias = 1, False
dilation = 1

Test_block = ConvolutionalProcessingBlock_BNRC(input_shape, filters, kernel_size, padding, bias, dilation)
test_input = torch.randn(input_shape)

# Test for correct shape
test_output = Test_block.forward(test_input)
if not test_output.shape == test_input.shape:
    all_passed = False
    print('Incorrect shape')

# Test for Batch Normalization mean 0 and standard deviation 1
out_conv0 = Test_block.layer_dict['conv_0'].forward(test_input)
out_bn0 = Test_block.layer_dict['bn_0'].forward(out_conv0)
if not np.allclose(out_bn0.mean().detach().numpy(),0):
    all_passed = False
    print('Mean is not 0 for bn_0')
if not (out_bn0.std().detach().numpy() > 0.999 and out_bn0.std().detach().numpy() < 1.001):
    all_passed = False
    print('Standard deviation is not 1 for bn_0')

out_relu = F.leaky_relu(out_bn0)
out_conv1 = Test_block.layer_dict['conv_1'].forward(out_relu)
out_bn1 = Test_block.layer_dict['bn_1'].forward(out_conv1)
if not np.allclose(out_bn1.mean().detach().numpy(),0):
    all_passed = False
    print('Mean is not 0 for bn_1')
elif not (out_bn1.std().detach().numpy() > 0.999 and out_bn1.std().detach().numpy() < 1.001 ):
    all_passed = False
    print('Standard deviation is not 1 for bn_1')

# Test for Residual Connection
if not torch.all(torch.eq(test_output, F.leaky_relu(out_bn1 + test_input))):
    all_passed = False
    print('Residual Connection is not implemented correctly')

# If all tests passed
if all_passed:
    print('All tests passed')