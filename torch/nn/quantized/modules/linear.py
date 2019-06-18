import math

import torch
from ...modules.module import Module
from ...._jit_internal import weak_module



@weak_module
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        quantized_weights: Quantized weight tensor of shape
            :math:`(\text{out\_features}, \text{in\_features})`
        quantized_bias: Quantized bias of the module of shape :math:`(\text{out\_features})`


    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:

        bias:   the quantized bias of the module of shape :math:`(\text{out\_features})`.


    Examples::

        >>> input_channels = 10
        >>> output_channels = 20
        >>> batch_size = 5
        >>> W_q = torch.rand(20, 10).float().quantize_linear(0.1, 4, torch.qint8)
        >>> B_q = torch.rand(output_channels).int()
        >>> input_q = torch.rand(batch_size, input_channels).float().quantize_linear(0.5,3,torch.qint8)
        >>> out_scale = 0.5
        >>> out_zero_point = 3
        >>> qLinear = nn.quantized.Linear(W_q, B_q, output_scale=out_scale, output_zero_point=out_zero_point)
        >>> output_q = qLinear(input_q)
        >>>print(output_q.size())
        torch.Size([5, 20])
    """

# TODO: Need to implement getstate and setstate functions, will be done in a later PR

    def __init__(self, quantized_weight, quantized_bias=None, output_scale=1, output_zero_point=0):
        super(Linear, self).__init__()

        #self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(quantized_weight)
        if quantized_bias is not None:
            self.bias = quantized_bias
        else:
            self.bias = torch.zeros(quantized_weight.shape[0]).float().quantize_linear(output_scale * quantized_weight.q_scale(),
                        0, torch.qint32)

        self.register_buffer('output_scale', torch.Tensor([output_scale]))
        self.register_buffer('output_zero_point', torch.Tensor([output_zero_point]))

    def forward(self, x):
        Y_q = torch.ops.quantized.fbgemm_linear(x, self._packed_weight, self.bias, self.output_scale, self.output_zero_point)
        return Y_q
