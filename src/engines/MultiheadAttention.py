import math
from typing import Union, Optional
import mindspore
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.ops.function.nn_func import multi_head_attention_forward
from mindspore.nn.cell import Cell
# from .basic import Dense, Dropout
# from .activation import ReLU, GELU
# from .normalization import LayerNorm
# from .container import CellList

class MultiheadAttention(Cell):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector, the key vector and value vector,
    the attention will be performed as the following:

    .. math::
        MultiHeadAttention(query, key, value) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`, and :math:`W^O` , :math:`W_i^Q` , :math:`W_i^K` ,
    :math:`W_i^V` are weight matrices. The default input / output projection layers is with a bias.

    if query, key and value tensor is same, then it will be self attention.

    Args:
        embed_dim (int): Total dimension of MultiheadAttention.
        num_heads (int): Number of attention heads. Note that `embed_dim` will be split
            across `num_heads` (i.e. each head will have dimension `embed_dim // num_heads`).
        dropout (float): Dropout probability of `attn_output_weights`. Default: ``0.0``.
        has_bias (bool): Whether adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv (bool): Whether adds bias to the key and value sequences at axis=0. Default: ``False``.
        add_zero_attn (bool): Whether adds a new batch of zeros to the key and value sequences at axis=1.
            Default: ``False``.
        kdim (int): Total number of features for keys. Default: ``None`` (`kdim=embed_dim`).
        vdim (int): Total number of features for values. Default: ``None`` (`vdim=embed_dim`).
        batch_first (bool): If ``True``, then the input and output shape are :math:`(batch, seq, feature)` ,
            else :math:`(seq, batch, feature)` . Default: ``False``.
        dtype (:class:`mindspore.dtype`): Data type of Parameter. Default: ``mstype.float32`` .

    Inputs:
        - **query** (Tensor): The query embeddings. If `query` is unbatched, the shape is :math:`(L, E_q)`,
          otherwise the shape is :math:`(L, N, E_q)` when `batch_first=False` or :math:`(N, L, E_q)` when
          `batch_first=True` , where :math:`L`is the target sequence length, :math:`N` is the batch size,
          and :math:`E_q` is the query embedding dimension `embed_dim`. Supported types: float16, float32,
          float64. Queries are compared against key-value pairs to produce the output.
        - **key** (Tensor): The key embeddings. If `key` is unbatched, the shape is :math:`(S, E_k)`, otherwise
          the shape is :math:`(S, N, E_k)` when `batch_first=False` or :math:`(N, S, E_k)` when
          `batch_first=True` , where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_k` is the key embedding dimension `kdim`. Supported types: float16, float32, float64.
        - **value** (Tensor): The value embeddings. If `value` is unbatched, the shape is :math:`(S, E_v)`,
          otherwise the shape is :math:`(S, N, E_v)` when `batch_first=False` or :math:`(N, S, E_v)` when
          `batch_first=True` , where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_v` is the value embedding dimension `vdim`. Supported types: float16, float32, float64.
        - **key_padding_mask** (Tensor, optional): If specified, a mask of shape :math:`(N, S)` indicating which
          elements within `key` to ignore for the purpose of attention (i.e. treat as "padding").
          For unbatched `query`, shape should be :math:`(S)`. Binary and float masks are supported.
          For a binary mask, a ``True`` value indicates that the corresponding `key` value will be ignored for
          the purpose of attention. For a float mask, it will be directly added to the corresponding `key` value.
          Supported float types: float16, float32, float64. Default: ``None``.
        - **need_weights** (bool): Whether returns `attn_output_weights` in addition to `attn_outputs`.
          Default: ``True``.
        - **attn_mask** (Tensor, optional): If specified, a 2D or 3D mask preventing attention to certain positions.
          Must be of shape :math:`(L, S)` or :math:`(N\cdot\text{num_heads}, L, S)`, where :math:`N` is the
          batch size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length.
          A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry
          in the batch. For a binary mask, a ``True`` value indicates that the corresponding position is not allowed
          to attend. For a float mask, the mask values will be added to the attention weight.
          Supported float types: float16, float32, float64. Default: ``None``.
        - **average_attn_weights** (bool): If true, indicates that the returned `attn_weights` should be averaged
          across heads. Otherwise, `attn_weights` are provided separately per head. Note that this flag only
          has an effect when `need_weights=True`. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        Tuple, a tuple contains(`attn_output`, `attn_output_weights`)

        - **attn_output** - Attention outputs. If input is unbatched, the output shape is :math:`(L, E)`, otherwise
          the output shape is :math:`(L, N, E)` when `batch_first=False` or :math:`(N, L, E)` when
          `batch_first=True` , where :math:`L` is the target sequence length, :math:`N` is the batch size,
          and :math:`E` is the embedding dimension `embed_dim`.
        - **attn_output_weights** - Only returned when `need_weights=True`. If `average_attn_weights=True`,
          returns attention weights averaged across heads with shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)` when input is batched, where :math:`N` is the batch size, :math:`L` is
          the target sequence length, and :math:`S` is the source sequence length.
          If `average_attn_weights=False`, returns attention weights per
          head of shape :math:`(\text{num_heads}, L, S)` when input is unbatched or
          :math:`(N, \text{num_heads}, L, S)` when input is batched.

    Raises:
        ValueError: If the init argument `embed_dim` is not divisible by `num_heads`.
        TypeError: If the input argument `key_padding_mask` is not bool or floating types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> embed_dim, num_heads = 128, 8
        >>> seq_length, batch_size = 10, 8
        >>> query = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
        >>> key = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
        >>> value = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
        >>> multihead_attn = ms.nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
        >>> print(attn_output.shape)
        (10, 8, 128)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, has_bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False, dtype=mstype.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("The init argument 'embed_dim' must be divisible by 'num_heads'.")

        if dtype is None:
            dtype = mindspore.float32
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, embed_dim), dtype), 'q_proj_weight')
            self.k_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.kdim), dtype), 'k_proj_weight')
            self.v_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.vdim), dtype), 'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * embed_dim, embed_dim), dtype),
                                            'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if has_bias:
            self.in_proj_bias = Parameter(initializer('zeros', (3 * embed_dim), dtype), 'in_proj_bias')
        else:
            self.in_proj_bias = None
        fan_in, _ = _calculate_fan_in_and_fan_out((embed_dim, embed_dim))
        bound = 1 / math.sqrt(fan_in)
        self.out_proj = Dense(embed_dim, embed_dim, has_bias=has_bias, weight_init=HeUniform(math.sqrt(5)),
                              bias_init=Uniform(bound), dtype=dtype)

        if add_bias_kv:
            self.bias_k = Parameter(initializer(XavierNormal(), (1, 1, embed_dim), dtype), 'bias_k')
            self.bias_v = Parameter(initializer(XavierNormal(), (1, 1, embed_dim), dtype), 'bias_v')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.k_is_v = False
        self.q_is_k = False
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query', args[0])
        key = kwargs.get('key', args[1])
        value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                  need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True):
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != mindspore.bool_ and not ops.is_floating_point(key_padding_mask):
                raise ValueError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # k_is_v and q_is_k preprocess in __call__ since Graph mode do not support `is`
            if self.k_is_v:
                if self.q_is_k:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query, key = [x.swapaxes(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.swapaxes(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k, dtype=self.dtype)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k, dtype=self.dtype)

        if self.batch_first and is_batched:
            attn_output = attn_output.swapaxes(1, 0)
        if need_weights:
            return attn_output, attn_output_weights
        return (attn_output,)