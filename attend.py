from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange


# constants
# 定义 EfficientAttentionConfig 命名元组，用于配置高效注意力机制
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
"""
EfficientAttentionConfig 命名元组用于配置高效注意力机制的不同实现方式。

参数:
    enable_flash (bool): 是否启用 FlashAttention。FlashAttention 是一种优化后的注意力计算方法，通常在 GPU 上具有更高的效率。
    enable_math (bool): 是否启用数学优化后的注意力计算方法。
    enable_mem_efficient (bool): 是否启用内存高效（Memory-Efficient）的注意力计算方法。这种方法通过减少内存占用，适用于处理大规模数据。
"""


# helpers

def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    参数:
        val: 需要检查的值。

    返回:
        bool: 如果值存在（不为 None），则返回 True；否则返回 False。
    """
    return val is not None


def once(fn):
    """
    创建一个装饰器，确保被装饰的函数只会被调用一次。

    参数:
        fn (function): 需要被限制只调用一次的函数。

    返回:
        function: 装饰后的函数。
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 使用 once 装饰器创建一个只打印一次的 print 函数
print_once = once(print)


# main class

class Attend(nn.Module):
    """
    注意力模块，用于计算输入序列的注意力权重和输出。

    参数:
        causal (bool, 可选): 是否使用因果掩码（causal mask）。默认值为 False。
        dropout (float, 可选): Dropout 概率。默认值为 0.0。
        flash (bool, 可选): 是否使用 FlashAttention 优化注意力计算。默认值为 False。
    """
    def __init__(
        self,
        causal = False,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        # Dropout 概率
        self.dropout = dropout
        # Dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 是否使用因果掩码
        self.causal = causal
        # 是否使用 FlashAttention
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        # 确定在 CUDA 和 CPU 上的高效注意力配置
        # CPU 上启用所有高效注意力配置
        self.cpu_config = EfficientAttentionConfig(True, True, True)
        # 初始化 CUDA 配置为 None
        self.cuda_config = None

        # 如果没有 CUDA 设备或不使用 FlashAttention，则不设置 CUDA 配置
        if not torch.cuda.is_available() or not flash:
            return

        # 获取 CUDA 设备属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # 如果是 A100 GPU，打印提示信息
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            # A100 上启用 FlashAttention，禁用数学和内存高效配置
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            # 如果不是 A100 GPU，打印提示信息
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            # 非 A100 上启用数学和内存高效配置，禁用 FlashAttention
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, i, j, device):
        """
        生成上三角掩码。

        参数:
            i (int): 序列长度。
            j (int): 另一个序列长度。
            device (torch.device): 张量所在的设备。

        返回:
            Tensor: 上三角掩码张量。
        """
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def flash_attn(self, q, k, v, mask = None, attn_bias = None):
        """
        使用 FlashAttention 计算注意力输出。

        参数:
            q (Tensor): 查询张量，形状为 (batch_size, heads, q_len, d_k)。
            k (Tensor): 键张量，形状为 (batch_size, heads, k_len, d_k)。
            v (Tensor): 值张量，形状为 (batch_size, heads, v_len, d_v)。
            mask (Tensor, 可选): 掩码张量，形状为 (batch_size, heads, q_len, k_len)。
            attn_bias (Tensor, 可选): 注意力偏置张量。

        返回:
            Tensor: 注意力输出张量。
        """
        # 解包张量形状信息
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # single headed key / values
        # 处理单头键和值

        if k.ndim == 3:
            # 重塑键张量形状
            k = rearrange(k, 'b n d -> b 1 n d')

        if v.ndim == 3:
            # 重塑值张量形状
            v = rearrange(v, 'b n d -> b 1 n d')

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        # 检查掩码是否存在并扩展到兼容的形状
        # 掩码是 B L，所以必须扩展到 B H N L

        if exists(mask) and mask.ndim != 4:
            # 重塑掩码张量形状
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # 扩展掩码张量形状
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # 检查是否有兼容的设备用于 FlashAttention

        # 选择配置
        config = self.cuda_config if is_cuda else self.cpu_config

        # 获取因果掩码标志
        causal = self.causal

        # handle attention bias
        # 处理注意力偏置

        if exists(attn_bias):
            # 计算掩码值
            mask_value = -torch.finfo(q.dtype).max // 2

            if causal:
                # 生成因果掩码
                causal_mask = self.get_mask(q_len, k_len, device)
                # 应用因果掩码
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value)

            if exists(mask):
                # 应用掩码
                attn_bias = attn_bias.masked_fill(~mask, mask_value)

            # 将注意力偏置作为掩码
            mask = attn_bias
            # 禁用因果掩码
            causal = False

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask, # 注意力掩码
                dropout_p = self.dropout if self.training else 0.,  # Dropout 概率
                is_causal = causal  # 是否使用因果掩码
            )
        # 返回注意力输出
        return out

    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        """
        前向传播方法，用于计算注意力输出。

        参数:
            q (Tensor): 查询张量，形状为 (batch_size, heads, q_len, d_k)。
            k (Tensor): 键张量，形状为 (batch_size, heads, k_len, d_k)。
            v (Tensor): 值张量，形状为 (batch_size, heads, v_len, d_v)。
            mask (Tensor, 可选): 掩码张量，形状为 (batch_size, q_len)。
            attn_bias (Tensor, 可选): 注意力偏置张量。

        Einstein 表示法:
            b - batch (批次)
            h - heads (注意力头)
            n, i, j - sequence length (序列长度，基础序列长度，源，目标)
            d - feature dimension (特征维度)
        """
        # 获取查询和键的长度，以及设备信息
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        # 计算缩放因子，通常为特征维度的平方根
        scale = q.shape[-1] ** -0.5
        # 根据键的张量维度选择 Einstein 求和约定
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if self.flash:
            assert not exists(attn_bias)
            # 使用 FlashAttention 计算注意力
            return self.flash_attn(q, k, v, mask = mask)

        # similarity
        # 计算相似度矩阵
        # 计算查询和键的点积，并缩放
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias
        # 添加注意力偏置
        if exists(attn_bias):
            # 将注意力偏置加到相似度矩阵上
            sim = sim + attn_bias

        # causal mask
        # 应用因果掩码
        if self.causal:
            # 生成因果掩码
            causal_mask = self.get_mask(q_len, k_len, device)
            # 应用因果掩码
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # key padding mask
        # 应用键填充掩码
        if exists(mask):
            if mask.ndim != 4:
                # 重塑掩码张量形状
                mask = rearrange(mask, 'b j -> b 1 1 j')
            # 应用掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重
        # 对相似度矩阵进行 softmax 归一化
        attn = sim.softmax(dim=-1)
        # 应用 Dropout
        attn = self.attn_dropout(attn)

        # aggregate values
        # 聚合值
        # 计算注意力输出
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)
        
        # 返回注意力输出
        return out
