"""Create ll5914_lab_2.ipynb from the template with all coding TODOs completed."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
nb = json.loads((root / 'Lab2_quantization.ipynb').read_text())

HDR = "    ##############################################################################\n"
END = ("    ##############################################################################\n"
       "    #END OF YOUR CODE                              #\n"
       "    ##############################################################################\n")

CODE = {}

CODE['def tensor_wise_quantize('] = '''###############################################
# Quantization Function Implementations
###############################################

import torch

def tensor_wise_quantize(x: torch.Tensor, num_bits: int) -> torch.Tensor:
''' + HDR + "    #TODO\n" + HDR + '''    # step 1: For simplicity, you can assume the quantization clipping range matches the maximum and minimum values exactly, no clipping is necessary.
    x_min, x_max = x.min(), x.max()
    q_max = 2 ** num_bits - 1
    # step 2: Calculate the scale s
    scale = (x_max - x_min) / q_max
    scale = scale.masked_fill(scale == 0, 1.0)
    # step 3: Calculate the fixed-point representation x_{int}
    x_int = torch.clamp(torch.round((x - x_min) / scale), 0, q_max)
    # step 4: Rescale
    x_quant = x_int * scale + x_min
''' + END + "    return x_quant"

CODE['def filter_wise_quantize('] = '''def filter_wise_quantize(x: torch.Tensor, num_bits: int, channel_axis: int = 0) -> torch.Tensor:
''' + HDR + "    #TODO\n" + HDR + '''    # step 1: Assume the quantization clipping range matches the maximum and minimum values exactly, no clipping is necessary.
    reduce_dims = [d for d in range(x.dim()) if d != channel_axis]
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    q_max = 2 ** num_bits - 1
    # step 2: Calculate the scale s, the scale will be different for each weight filter
    scale = (x_max - x_min) / q_max
    scale = scale.masked_fill(scale == 0, 1.0)
    # step 3: Calculate the fixed-point representation x_{int}
    x_int = torch.clamp(torch.round((x - x_min) / scale), 0, q_max)
    # step 4: Rescale
    x_quant = x_int * scale + x_min
''' + END + "    return x_quant"

CODE['def tensor_wise_quantize_with_factor_symmetric('] = '''def tensor_wise_quantize_with_factor_symmetric(x: torch.Tensor,num_bits: int,clipping_factor: float = 1.0):
''' + HDR + "    #TODO\n" + HDR + '''    # step 1: Using the clipping factor to compute the clipping range, then truncate x using the clipping rate
    clip_range = clipping_factor * x.abs().max()
    x_clipped = torch.clamp(x, -clip_range, clip_range)
    # step 2: Calculate the scale s
    scale = 2 * clip_range / (2 ** num_bits - 2)
    scale = scale.masked_fill(scale == 0, 1.0)
    # step 3: Calculate the fixed-point representation x_{int}
    x_int = torch.round(x_clipped / scale)
    # step 4: Rescale
    x_quant = x_int * scale
''' + END + "\n    return x_quant"

CODE['def compute_s_e_m('] = '''import torch
def compute_s_e_m(x: torch.Tensor, bias: int = 127):
    """
    Given a FP32 tensor x, compute the sign(s), exponent (e), and mantissa (m) according to:
      1) a = floor(log2(|x|))
      2) e = a + bias
      3) m = |x| / 2^a - 1
    Note: This ignores the sign bit (s) for now and assumes x != 0.

    :param x: A torch.Tensor of float32 values
    :param bias: The bias used in the exponent, default is 127 for FP32
    :return: (e, m) where both are torch.Tensor
    """
''' + HDR + "    #TODO\n" + HDR + '''    # step 1: Generate the sign matrix of x and compute the absolute value of x
    s = (x < 0).to(torch.int32)
    x_abs = x.abs()
    # step 2: a = floor(log2(|x|))
    a = torch.floor(torch.log2(x_abs))
    # step 3: e = a + bias
    e = (a + bias).to(torch.int32)
    # step 4: m = |x| / 2^a - 1
    m = x_abs / torch.pow(2.0, a) - 1
''' + END + "    return s, e, m"

CODE['def tensor_wise_quantize_with_factor_symmetric_ste('] = '''###############################################
# STE Wrapper Function (for Quantization-Aware Training)
###############################################
def tensor_wise_quantize_with_factor_symmetric_ste(x: torch.Tensor,num_bits: int,clipping_factor: float = 1.0):
''' + HDR + "    #TODO: Conduct STE to round operation\n" + HDR + '''    # step 1: Using the clipping factor to compute the clipping range, then truncate x using the clipping rate
    clip_range = clipping_factor * x.detach().abs().max()
    x_clipped = torch.clamp(x, -clip_range, clip_range)
    # step 2: Calculate the scale s
    scale = 2 * clip_range / (2 ** num_bits - 2)
    scale = scale.masked_fill(scale == 0, 1.0)
    # step 3: Calculate the fixed-point representation x_{int}, with STE enabled
    x_scaled = x_clipped / scale
    x_int = x_scaled + (torch.round(x_scaled) - x_scaled).detach()
    # step 4: Rescale
    x_quant = x_int * scale
''' + "    ##############################################################################\n    #END OF YOUR CODE                             #\n    ##############################################################################\n\n    return x_quant"

LM_BODY = '''    # step 1: Assume the quantization clipping range matches the maximum and minimum values exactly, no clipping is necessary.
{min_max}    q_max = 2 ** num_bits - 1
    # step 2: {step2}
    scale = (x_max - x_min) / q_max
    scale = scale.masked_fill(scale == 0, 1.0)
    # step 3: Calculate the fixed-point representation x_{{int}}
    x_int = torch.clamp(torch.round((x - x_min) / scale), 0, q_max)
    # step 4: Rescale
    x_quant = x_int * scale + x_min
'''

CODE['def lm_tensor_wise_quantize('] = '''###############################################
# Quantization Function Implementations
###############################################

def lm_tensor_wise_quantize(x: torch.Tensor, num_bits: int) -> torch.Tensor:
''' + HDR + "    #TODO: implement tensor-wise quantization for linear layer\n" + HDR + LM_BODY.format(
    min_max="    x_min, x_max = x.min(), x.max()\n", step2="Calculate the scale s") + END + "    return x_quant"

CODE['def lm_channel_wise_quantize('] = '''def lm_channel_wise_quantize(x: torch.Tensor, num_bits: int) -> torch.Tensor:
''' + HDR + "    #TODO: implement channel-wise quantization for linear layer\n" + HDR + LM_BODY.format(
    min_max=("    # nn.Linear weight is [out_features, in_features]; each output channel is one row\n"
             "    x_min = x.amin(dim=-1, keepdim=True)\n    x_max = x.amax(dim=-1, keepdim=True)\n"),
    step2="Figure out the input shapes and calculate the scale s, the scale will be different for each channel") + END + '''    return x_quant


def lm_token_wise_quantize(x: torch.Tensor, num_bits: int) -> torch.Tensor:
''' + HDR + "    #TODO: implement token-wise quantization for linear layer\n" + HDR + LM_BODY.format(
    min_max=("    # activations are [batch, tokens, features]; each token is one vector along the last dim\n"
             "    x_min = x.amin(dim=-1, keepdim=True)\n    x_max = x.amax(dim=-1, keepdim=True)\n"),
    step2="Figure out the input shapes and compute the scale s, the scale will be different for each token") + END + "    return x_quant"

CODE['def lm_group_wise_quantize('] = '''def lm_group_wise_quantize(x: torch.Tensor, num_bits: int, group_size: int = 128) -> torch.Tensor:
''' + HDR + "    #TODO: implement group-wise quantization for the weights of linear layer\n" + HDR + '''    # step 1: Split each row of the weight [out_features, in_features] into groups of `group_size` consecutive elements
    out_features, in_features = x.shape
    x_grouped = x.reshape(out_features, in_features // group_size, group_size)
    # step 2: Calculate the min, max and scale s for each group
    x_min = x_grouped.amin(dim=-1, keepdim=True)
    x_max = x_grouped.amax(dim=-1, keepdim=True)
    q_max = 2 ** num_bits - 1
    scale = (x_max - x_min) / q_max
    scale = scale.masked_fill(scale == 0, 1.0)
    # step 3: Calculate the fixed-point representation x_{int}
    x_int = torch.clamp(torch.round((x_grouped - x_min) / scale), 0, q_max)
    # step 4: Rescale and reshape back to [out_features, in_features]
    x_quant = (x_int * scale + x_min).reshape(out_features, in_features)
''' + END + "    return x_quant"


LAYER_FACTORS = {'conv1': 1.0, 'linear': 1.0}
for blk in ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']:
    LAYER_FACTORS[f'{blk}.conv1'] = 0.85
    LAYER_FACTORS[f'{blk}.conv2'] = 0.85
    if blk.endswith('.0') and blk != 'layer1.0':
        LAYER_FACTORS[f'{blk}.shortcut.0'] = 0.9
ORDER = ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
         'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.shortcut.0', 'layer2.1.conv1', 'layer2.1.conv2',
         'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.shortcut.0', 'layer3.1.conv1', 'layer3.1.conv2',
         'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.shortcut.0', 'layer4.1.conv1', 'layer4.1.conv2', 'linear']
CODE['layer_dict = {'] = (
    "###########################################################################################\n"
    "#TODO: Please select layer-wise clipping factors for weights to reach the optimal performance\n"
    "###########################################################################################\n"
    "layer_dict = {\n" + ",\n".join(f"    '{k}': {LAYER_FACTORS[k]}" for k in ORDER) + "\n}\n"
    "##############################################################################\n"
    "#END OF YOUR CODE                              #\n"
    "##############################################################################")


def find(marker):
    idx = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
    assert len(idx) == 1, (marker, idx)
    return idx[0]


for marker, src in CODE.items():
    nb['cells'][find(marker)]['source'] = src.splitlines(True)

out = root / 'll5914_lab_2.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print('wrote', out)
