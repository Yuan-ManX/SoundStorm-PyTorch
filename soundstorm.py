from __future__ import annotations

import math
from random import random, randrange
from functools import wraps
from contextlib import nullcontext
from collections import namedtuple
from pathlib import Path

from tqdm import tqdm

import torch
from torch.amp import autocast
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, unpack, pack
from einops.layers.torch import Rearrange, EinMix

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Any

from spear_tts_pytorch import TextToSemantic
from audiolm_pytorch import SoundStream
from audiolm_pytorch import HubertWithKmeans, FairseqVQWav2Vec
from gateloop_transformer import SimpleGateLoopLayer as GateLoop

from attend import Attend


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


def default(val, d):
    """
    如果值存在，则返回该值；否则返回默认值。

    参数:
        val: 需要检查的可选值。
        d: 默认值。

    返回:
        Any: 如果 val 存在，则返回 val；否则返回 d。
    """
    return val if exists(val) else d


def divisible_by(numer, denom):
    """
    检查一个数是否可以被另一个数整除。

    参数:
        numer (int): 被除数。
        denom (int): 除数。

    返回:
        bool: 如果 numer 可以被 denom 整除，则返回 True；否则返回 False。
    """
    return (numer % denom) == 0


def calc_same_padding(kernel_size):
    """
    计算用于保持输入和输出尺寸相同的填充大小。

    参数:
        kernel_size (int): 卷积核的大小。

    返回:
        Tuple[int, int]: 填充大小，通常为 (pad, pad) 或 (pad, pad - 1)。
    """
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def eval_decorator(fn):
    """
    创建一个装饰器，用于在评估模式下运行函数，并在函数执行前后保持模型的训练状态。

    参数:
        fn (function): 需要装饰的函数。

    返回:
        function: 装饰后的函数。
    """
    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


# sampling helpers

def top_k(logits, thres = 0.9):
    """
    对输入的 logits 应用 top-k 过滤。

    参数:
        logits (Tensor): 输入的 logits 张量。
        thres (float, 可选): 保留 top-k 的阈值，范围在 0 到 1 之间。默认值为 0.9。

    返回:
        Tensor: 应用 top-k 过滤后的 logits 张量。
    """
    # 计算要保留的 top-k 值
    k = math.ceil((1 - thres) * logits.shape[-1])
    # 获取 top-k 的值和索引
    val, ind = logits.topk(k, dim = -1)
    # 创建一个与 logits 形状相同的张量，并用负无穷填充
    probs = torch.full_like(logits, float('-inf'))
    # 将 top-k 的值填充回 probs 张量
    probs.scatter_(2, ind, val)
    # 返回应用 top-k 过滤后的 probs 张量
    return probs


def log(t, eps = 1e-10):
    """
    计算输入张量的对数，并添加一个极小值以防止数值不稳定。

    参数:
        t (Tensor): 输入张量。
        eps (float, 可选): 添加的极小值，防止 log(0) 导致的数值不稳定。默认值为 1e-10。

    返回:
        Tensor: 输入张量的对数结果。
    """
    return torch.log(t + eps)


def gumbel_noise(t):
    """
    生成与输入张量形状相同的 Gumbel 噪声。

    参数:
        t (Tensor): 输入张量，用于确定噪声的形状。

    返回:
        Tensor: 与输入张量形状相同的 Gumbel 噪声。
    """
    # 生成均匀分布的噪声，范围在 0 到 1 之间
    noise = torch.zeros_like(t).uniform_(0, 1)
    # 应用 Gumbel 变换生成 Gumbel 噪声
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim = -1):
    """
    对输入张量应用 Gumbel-Softmax 采样。

    参数:
        t (Tensor): 输入张量。
        temperature (float, 可选): 温度参数，控制采样的平滑程度。默认值为 1.0。
        dim (int, 可选): 采样的维度。默认值为 -1。

    返回:
        Tensor: 采样后的张量。
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


# prob helpers
# 概率辅助函数

def sample_prob(prob):
    """
    根据给定的概率进行采样。

    参数:
        prob (float): 采样概率。

    返回:
        bool: 如果随机数小于等于概率，则返回 True；否则返回 False。
    """
    return random() < prob


def coin_flip():
    """
    进行一次硬币抛掷采样（50% 的概率）。

    返回:
        bool: 返回 True 或 False，概率各为 50%。
    """
    return sample_prob(0.5)


# tensor helpers

@beartype
def get_mask_subset_prob(
    mask: Tensor,
    prob: float | Tensor,
    min_mask: int = 0,
    min_keep_mask: int = 0
):
    """
    根据给定的概率和约束条件，从输入的掩码中随机选择子集。

    参数:
        mask (Tensor): 输入的掩码张量，形状为 (batch_size, sequence_length)。
        prob (float 或 Tensor): 掩码的概率，可以是浮点数或形状为 (batch_size,) 的张量。
        min_mask (int, 可选): 每个样本的最小掩码数量。默认值为 0。
        min_keep_mask (int, 可选): 每个样本的最小保留掩码数量。默认值为 0。

    返回:
        Tensor: 生成的子集掩码张量，形状为 (batch_size, sequence_length)。
    """
    # 获取批大小、序列长度和设备信息
    batch, seq, device = *mask.shape, mask.device

    if isinstance(prob, Tensor):
        # 如果 prob 是张量，则重塑为 (batch_size, 1)
        prob = rearrange(prob, 'b -> b 1')
    
    # 计算每个样本中需要掩码的总数量
    total = mask.sum(dim = -1, keepdim = True)
    
    # 计算每个样本中最多可以掩码的数量
    max_mask = (total - min_keep_mask).clamp(min = 0)
    
    # 计算需要掩码的数量，并确保不小于 min_mask
    num_to_mask = (total * prob).long().clamp(min = min_mask)
    # 确保需要掩码的数量不超过最大允许的掩码数量
    num_to_mask = torch.minimum(num_to_mask, max_mask)

    # 生成随机的 logits 张量，范围在 [0, 1) 之间
    logits = torch.rand((batch, seq), device = device)
    # 将不需要掩码的位置填充为 -1
    logits = logits.masked_fill(~mask, -1)

    # 对 logits 进行排序，获取排序后的索引，并转换为浮点数
    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    # 计算每个样本中填充的数量
    num_padding = (~mask).sum(dim = -1, keepdim = True)
    # 调整排序索引以排除填充位置
    randperm -= num_padding

    # 根据需要掩码的数量生成子集掩码
    subset_mask = randperm < num_to_mask
    # 确保不需要掩码的位置保持为 False
    subset_mask.masked_fill_(~mask, False)
    # 返回生成的子集掩码
    return subset_mask


# schedules

def linear_schedule(t):
    """
    线性调度函数。

    参数:
        t (float): 当前时间步长，范围在 [0, 1] 之间。

    返回:
        float: 调度后的值，范围在 [0, 1] 之间。
    """
    # 返回线性递减的值
    return 1 - t


def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    """
    余弦调度函数。

    参数:
        t (float): 当前时间步长，范围在 [0, 1] 之间。

    返回:
        float: 调度后的值，范围在 [0, 1] 之间。
    """
    # 返回余弦递减的值
    return torch.cos(t * math.pi / 2)


# rotary embedding

class RotaryEmbedding(Module):
    """
    旋转位置编码（Rotary Position Embedding）模块。
    该模块用于在自注意力机制中引入位置信息，通过旋转输入向量来实现。

    参数:
        dim (int): 输入特征的维度。
        theta (float, 可选): 控制频率的缩放因子。默认值为 10000。
    """
    def __init__(self, dim, theta = 10000):
        super().__init__()
        # 计算逆频率，用于生成旋转角度
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        # 注册逆频率缓冲区，不作为模型参数保存
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        """
        获取当前设备信息。

        返回:
            torch.device: 当前设备（CPU 或 GPU）。
        """
        return next(self.buffers()).device

    @autocast('cuda', enabled = False)
    def forward(self, seq_len):
        """
        生成旋转位置编码。

        参数:
            seq_len (int): 序列长度。

        返回:
            Tensor: 旋转位置编码张量，形状为 (seq_len, dim)。
        """
        # 生成位置索引张量
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        # 计算频率张量
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        # 将频率张量复制并拼接，以匹配输入特征的维度
        freqs = torch.cat((freqs, freqs), dim = -1)
        # 返回旋转位置编码
        return freqs


def rotate_half(x):
    """
    对输入张量的后半部分进行旋转。

    参数:
        x (Tensor): 输入张量，形状为 (..., dim)。

    返回:
        Tensor: 旋转后的张量，形状为 (..., dim)。
    """
    # 将输入张量拆分为两部分
    x1, x2 = x.chunk(2, dim=-1)
    # 将后半部分取反并与前半部分拼接，实现旋转
    return torch.cat((-x2, x1), dim=-1)


@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(pos, t):
    """
    应用旋转位置编码到输入张量。

    参数:
        pos (Tensor): 旋转位置编码张量，形状为 (seq_len, dim)。
        t (Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。

    返回:
        Tensor: 应用旋转位置编码后的张量，形状为 (batch_size, seq_len, dim)。
    """
    # 应用旋转位置编码
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# t5 relative positional bias

class T5RelativePositionBias(Module):
    """
    T5 相对位置偏置模块，用于在 Transformer 模型中引入相对位置信息。
    该模块通过桶化（bucketing）方法将相对位置映射到不同的桶中，并学习每个桶的偏置。

    参数:
        scale (float, 可选): 偏置的缩放因子。默认值为 1.0。
        num_buckets (int, 可选): 桶的数量。默认值为 32。
        max_distance (int, 可选): 最大相对距离，超过此距离的相对位置将被映射到同一个桶中。默认值为 128。
        heads (int, 可选): 注意力头的数量。默认值为 8。
    """
    def __init__(
        self,
        scale = 1.,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        # 保存缩放因子
        self.scale = scale
        # 保存桶的数量
        self.num_buckets = num_buckets
        # 保存最大相对距离
        self.max_distance = max_distance
        # 定义嵌入层，用于学习每个桶的偏置
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        """
        将相对位置映射到不同的桶中。

        参数:
            relative_position (Tensor): 输入的相对位置张量。
            num_buckets (int, 可选): 桶的数量。默认值为 32。
            max_distance (int, 可选): 最大相对距离。默认值为 128。

        返回:
            Tensor: 映射后的桶索引张量。
        """
        # 初始化返回值
        ret = 0
        # 取相对位置的负值
        n = -relative_position

        # 将桶的数量减半
        num_buckets //= 2
        # 如果相对位置小于 0，则加上桶数量的一半
        ret += (n < 0).long() * num_buckets
        # 取绝对值
        n = torch.abs(n)

        # 计算精确桶的最大索引
        max_exact = num_buckets // 2
        # 判断相对位置是否小于精确桶的最大索引
        is_small = n < max_exact

        # 计算大于精确桶的相对位置对应的桶索引
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(
            val_if_large,
            # 确保桶索引不超过最大桶索引
            torch.full_like(val_if_large, num_buckets - 1)
        )

        # 根据相对位置的大小选择桶索引
        ret += torch.where(is_small, n, val_if_large)
        # 返回映射后的桶索引
        return ret

    @property
    def device(self):
        """
        获取当前设备信息。

        返回:
            torch.device: 当前设备（CPU 或 GPU）。
        """
        return next(self.parameters()).device

    def forward(self, n):
        """
        前向传播方法，用于计算相对位置偏置。

        参数:
            n (int): 序列长度。

        返回:
            Tensor: 相对位置偏置张量，形状为 (heads, n, n)。
        """
        # 生成位置索引张量
        pos = torch.arange(n, device = self.device).long()
        # 计算相对位置张量
        rel_pos = rearrange(pos, 'j -> 1 j') - rearrange(pos, 'i -> i 1')

        # 将相对位置映射到桶中
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        # 获取桶对应的偏置值
        values = self.relative_attention_bias(rp_bucket)

        # 重塑偏置张量的形状
        bias = rearrange(values, 'i j h -> h i j')
        # 返回缩放后的偏置
        return bias * self.scale


# conformer

class Swish(Module):
    """
    Swish 激活函数模块。
    Swish 是一种自门控激活函数，定义为 x * sigmoid(x)。

    参数:
        无
    """
    def forward(self, x):
        """
        前向传播方法，应用 Swish 激活函数。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 应用 Swish 激活后的张量。
        """
        return x * x.sigmoid()


class GLU(Module):
    """
    GLU（Gated Linear Unit）模块。
    GLU 是一种门控机制，通过将输入分成两部分，一部分作为门控信号，另一部分作为输出信号。

    参数:
        dim (int): 分割维度的索引。
    """
    def __init__(self, dim):
        super().__init__()
        # 保存分割维度的索引
        self.dim = dim

    def forward(self, x):
        """
        前向传播方法，应用 GLU 门控机制。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 应用 GLU 门控后的张量。
        """
        # 将输入张量分成两部分
        out, gate = x.chunk(2, dim=self.dim)
        # 应用 GLU 门控机制
        return out * gate.sigmoid()


class DepthWiseConv1d(Module):
    """
    深度可分离卷积1D模块。
    深度可分离卷积将标准卷积分解为深度卷积和逐点卷积，从而减少参数量和计算量。

    参数:
        chan_in (int): 输入通道数。
        chan_out (int): 输出通道数。
        kernel_size (int): 卷积核大小。
        padding (int): 填充大小。
    """
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        # 保存填充大小
        self.padding = padding
        # 定义深度可分离卷积层
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x, mask = None):
        """
        前向传播方法，应用深度可分离卷积。

        参数:
            x (Tensor): 输入张量。
            mask (Tensor, 可选): 掩码张量，用于掩码卷积操作。默认值为 None。

        返回:
            Tensor: 卷积后的张量。
        """
        if exists(mask):
            # 重塑掩码张量的形状
            mask = rearrange(mask, 'b n -> b 1 n')
            # 应用掩码
            x = x.masked_fill(~mask, 0.)

        # 对输入张量进行填充
        x = F.pad(x, self.padding)
        # 进行卷积操作
        out = self.conv(x)

        if exists(mask):
            # 再次应用掩码
            out = out.masked_fill(~mask, 0.)

        # 返回卷积后的张量
        return out


# attention, feedforward, and conv module

class Scale(Module):
    """
    Scale 模块，用于在函数输出上应用缩放因子。

    参数:
        scale (float): 缩放因子。
        fn (callable): 需要应用缩放因子的函数。
    """
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        """
        前向传播方法，应用函数并乘以缩放因子。

        参数:
            x (Tensor): 输入张量。
            **kwargs: 传递给函数的附加关键字参数。

        返回:
            Tensor: 应用函数并缩放后的张量。
        """
        return self.fn(x, **kwargs) * self.scale


class ChanLayerNorm(Module):
    """
    ChanLayerNorm 模块，用于对每个通道进行层归一化。

    参数:
        dim (int): 输入张量的通道维度。
    """
    def __init__(self, dim):
        super().__init__()
        # 定义可学习的缩放参数 gamma，形状为 (1, dim, 1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        """
        前向传播方法，应用通道层归一化。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 归一化后的张量。
        """
        # 根据数据类型设置极小值，防止数值不稳定
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        # 计算每个通道的方差
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        # 计算每个通道的均值
        mean = torch.mean(x, dim = 1, keepdim = True)
        # 应用通道层归一化
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.gamma


class PreNorm(Module):
    """
    PreNorm 模块，用于在函数应用之前对输入进行层归一化。

    参数:
        dim (int): 输入张量的特征维度。
        fn (callable): 需要应用归一化的函数。
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        """
        前向传播方法，应用层归一化并调用函数。

        参数:
            x (Tensor): 输入张量。
            **kwargs: 传递给函数的附加关键字参数。

        返回:
            Tensor: 应用函数后的张量。
        """
        # 应用层归一化
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(Module):
    """
    注意力机制模块，用于计算输入序列的注意力权重和输出。

    参数:
        dim (int): 输入特征的维度。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        dropout (float, 可选): Dropout 概率。默认值为 0.0。
        flash (bool, 可选): 是否使用 FlashAttention 优化注意力计算。默认值为 True。
    """
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 保存注意力头的数量
        self.heads= heads
        # 计算缩放因子
        self.scale = dim_head ** -0.5

        # 定义 Attend 模块，用于注意力计算
        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 定义线性变换层，用于生成查询 (Q)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 定义线性变换层，用于生成键 (K) 和值 (V)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        # 定义线性变换层，用于生成输出
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None,
        attn_bias = None,
        return_values = False,
        value_residual = None
    ):
        """
        前向传播方法，应用注意力机制。

        参数:
            x (Tensor): 输入张量。
            context (Tensor, 可选): 上下文张量，用于生成键和值。默认值为 None，表示使用输入张量作为上下文。
            mask (Tensor, 可选): 掩码张量，用于掩码注意力计算。默认值为 None。
            rotary_emb (Tensor, 可选): 旋转位置编码，用于位置感知注意力。默认值为 None。
            attn_bias (Tensor, 可选): 注意力偏置，用于调整注意力权重。默认值为 None。
            return_values (bool, 可选): 是否返回值张量。默认值为 False。
            value_residual (Tensor, 可选): 值残差，用于残差连接。默认值为 None。

        返回:
            Tensor: 注意力输出张量。如果 return_values 为 True，则返回元组 (输出张量, 值张量)。
        """
        # 获取序列长度、设备信息、注意力头数量以及是否存在上下文
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        # 如果没有提供上下文，则使用输入张量作为上下文
        context = default(context, x)

        # 生成查询 (Q)、键 (K) 和值 (V)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # 重塑张量形状
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(value_residual):
            # 如果存在值残差，则将其与值张量混合
            v = 0.5 * (v + value_residual)

        if exists(rotary_emb):
            # 应用旋转位置编码到查询
            q = apply_rotary_pos_emb(rotary_emb, q)
            # 应用旋转位置编码到键
            k = apply_rotary_pos_emb(rotary_emb, k)

        # 进行注意力计算
        out = self.attend(q, k, v, mask = mask, attn_bias = attn_bias)

        # 重塑输出张量形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 应用输出线性变换
        out = self.to_out(out)

        # 如果不需要返回值张量，则返回输出
        if not return_values:
            return out

        # 如果需要返回值张量，则返回输出和值张量
        return out, v


class FeedForward(Module):
    """
    前馈神经网络模块，用于对输入进行非线性变换。

    参数:
        dim (int): 输入特征的维度。
        mult (int, 可选): 隐藏层维度的乘法因子。默认值为 4。
        dropout (float, 可选): Dropout 概率。默认值为 0.0。
    """
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), # 线性变换层，将维度从 dim 增加到 dim * mult
            Swish(),  # Swish 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(dim * mult, dim),  # 线性变换层，将维度从 dim * mult 减少到 dim
            nn.Dropout(dropout)  # Dropout 层
        )

    def forward(self, x):
        """
        前向传播方法，应用前馈神经网络。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 应用前馈神经网络后的张量。
        """
        return self.net(x)


class ConformerConvModule(Module):
    """
    Conformer 卷积模块，用于在 Conformer 模型中引入卷积操作。

    参数:
        dim (int): 输入特征的维度。
        causal (bool, 可选): 是否使用因果卷积。默认值为 False。
        expansion_factor (int, 可选): 隐藏层维度的扩展因子。默认值为 2。
        kernel_size (int, 可选): 卷积核大小。默认值为 31。
        dropout (float, 可选): Dropout 概率。默认值为 0.0。
    """
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()
        # 计算内部维度
        inner_dim = dim * expansion_factor
        # 计算填充大小
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net1 = nn.Sequential(
            nn.LayerNorm(dim), # 层归一化
            Rearrange('b n c -> b c n'), # 重塑张量形状
            nn.Conv1d(dim, inner_dim * 2, 1), # 1D 卷积层，扩展维度
            GLU(dim=1) # GLU 门控机制
        )

        # 深度可分离卷积
        self.ds_conv = DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding)

        self.net2 = nn.Sequential(
            Swish(), # Swish 激活函数
            ChanLayerNorm(inner_dim), # 通道层归一化
            nn.Conv1d(inner_dim, dim, 1), # 1D 卷积层，恢复维度
            Rearrange('b c n -> b n c'), # 重塑张量形状
            nn.Dropout(dropout) # Dropout 层
        )

    def forward(self, x, mask = None):
        """
        前向传播方法，应用 Conformer 卷积模块。

        参数:
            x (Tensor): 输入张量。
            mask (Tensor, 可选): 掩码张量，用于掩码卷积操作。默认值为 None。

        返回:
            Tensor: 应用 Conformer 卷积模块后的张量。
        """
        # 应用第一个网络块
        x = self.net1(x)
        # 应用深度可分离卷积
        x = self.ds_conv(x, mask = mask)
        # 应用第二个网络块
        return self.net2(x)


# Conformer Block

class ConformerBlock(Module):
    """
    Conformer 块模块，结合了前馈神经网络、注意力机制和卷积操作。

    参数:
        dim (int): 输入特征的维度。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        ff_mult (int, 可选): 前馈神经网络隐藏层维度的乘法因子。默认值为 4。
        conv_expansion_factor (int, 可选): 卷积模块隐藏层维度的扩展因子。默认值为 2。
        conv_kernel_size (int, 可选): 卷积核大小。默认值为 31。
        attn_dropout (float, 可选): 注意力机制的 Dropout 概率。默认值为 0.0。
        attn_flash (bool, 可选): 是否使用 FlashAttention。默认值为 True。
        ff_dropout (float, 可选): 前馈神经网络的 Dropout 概率。默认值为 0.0。
        conv_dropout (float, 可选): 卷积模块的 Dropout 概率。默认值为 0.0。
        conv_causal (bool, 可选): 是否使用因果卷积。默认值为 False。
        use_gateloop_layers (bool, 可选): 是否使用门控循环层。默认值为 False。
    """
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        attn_flash = True,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        use_gateloop_layers = False
    ):
        super().__init__()
        # 定义第一个前馈神经网络
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # 定义门控循环层（如果启用）
        self.gateloop = GateLoop(dim) if use_gateloop_layers else None

        # 定义注意力机制
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash)
        # 定义卷积模块
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        
        # 定义第二个前馈神经网络
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # 对注意力机制应用前置归一化
        self.attn = PreNorm(dim, self.attn)
        # 对第一个前馈神经网络应用缩放和前置归一化
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # 对第二个前馈神经网络应用缩放和前置归一化
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        # 定义层归一化层
        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        attn_bias = None,
        attn_value_residual = None,
        return_values = False
    ):
        """
        前向传播方法，应用 Conformer 块。

        参数:
            x (Tensor): 输入张量。
            mask (Tensor, 可选): 掩码张量，用于掩码注意力计算。默认值为 None。
            rotary_emb (Tensor, 可选): 旋转位置编码，用于位置感知注意力。默认值为 None。
            attn_bias (Tensor, 可选): 注意力偏置，用于调整注意力权重。默认值为 None。
            attn_value_residual (Tensor, 可选): 注意力值残差，用于残差连接。默认值为 None。
            return_values (bool, 可选): 是否返回值张量。默认值为 False。

        返回:
            Tensor: 应用 Conformer 块后的张量。如果 return_values 为 True，则返回元组 (输出张量, 注意力值张量)。
        """
        # 应用第一个前馈神经网络并添加残差连接
        x = self.ff1(x) + x

        if exists(self.gateloop):
            # 应用门控循环层并添加残差连接
            x = self.gateloop(x) + x

        # 应用注意力机制并返回值
        attn_out, attn_values = self.attn(x, mask = mask, rotary_emb = rotary_emb, attn_bias = attn_bias, value_residual = attn_value_residual, return_values = True)
        # 添加注意力输出残差
        x = attn_out + x

        # 应用卷积模块并添加残差
        x = self.conv(x, mask = mask) + x
        # 应用第二个前馈神经网络并添加残差
        x = self.ff2(x) + x
        # 应用层归一化
        x = self.post_norm(x)

        if not return_values:
            # 如果不需要返回值张量，则返回输出
            return x
        
        # 如果需要返回值张量，则返回输出和注意力值
        return x, attn_values


# Conformer

class Conformer(Module):
    """
    Conformer 模型类，结合了卷积、注意力机制和前馈神经网络。

    参数:
        dim (int): 输入特征的维度。
        depth (int): Conformer 块的深度，即堆叠的 ConformerBlock 数量。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        ff_mult (int, 可选): 前馈神经网络隐藏层维度的乘法因子。默认值为 4。
        conv_expansion_factor (int, 可选): 卷积模块隐藏层维度的扩展因子。默认值为 2。
        conv_kernel_size (int, 可选): 卷积核大小。默认值为 31。
        attn_dropout (float, 可选): 注意力机制的 Dropout 概率。默认值为 0.0。
        ff_dropout (float, 可选): 前馈神经网络的 Dropout 概率。默认值为 0.0。
        conv_dropout (float, 可选): 卷积模块的 Dropout 概率。默认值为 0.0。
        conv_causal (bool, 可选): 是否使用因果卷积。默认值为 False。
        attn_flash (bool, 可选): 是否使用 FlashAttention。默认值为 True。
        t5_rel_pos_bias (bool, 可选): 是否使用 T5 相对位置偏置。默认值为 False。
        use_gateloop_layers (bool, 可选): 是否使用门控循环层。默认值为 True。
    """
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        attn_flash = True,
        t5_rel_pos_bias = False,
        use_gateloop_layers = True
    ):
        super().__init__()
        # 确保 FlashAttention 和 T5 相对位置偏置不同时使用
        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'

        # 保存输入特征的维度
        self.dim = dim
        # 初始化模块列表，用于存储 ConformerBlock
        self.layers = ModuleList([])

        # 定义旋转位置编码或 T5 相对位置偏置
        # 如果不使用 T5 相对位置偏置，则定义旋转位置编码
        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        # 如果使用 T5 相对位置偏置，则定义 T5 相对位置偏置
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None

        # 堆叠 ConformerBlock
        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal,
                attn_flash = attn_flash,
                use_gateloop_layers = use_gateloop_layers
            ))

    def forward(self, x, mask = None):
        """
        前向传播方法，应用 Conformer 模型。

        参数:
            x (Tensor): 输入张量。
            mask (Tensor, 可选): 掩码张量，用于掩码注意力计算。默认值为 None。

        返回:
            Tensor: 应用 Conformer 模型后的张量。
        """
        # 获取序列长度
        seq_len = x.shape[-2]

        # 生成旋转位置编码
        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        # 生成 T5 相对位置偏置
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None

        # 初始化注意力值残差
        attn_value_residual = None

        for block in self.layers:
            # 应用 ConformerBlock，并获取注意力输出和值
            x, attn_values = block(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias,
                attn_value_residual = attn_value_residual,
                return_values = True
            )
            # 更新注意力值残差
            attn_value_residual = default(attn_value_residual, attn_values)

        return x


# conformer with sum reduction across quantized tokens at the beginning, along with heads

class ConformerWrapper(Module):
    """
    ConformerWrapper 类，用于包装 Conformer 模型，并结合量化器（quantizers）进行操作。
    该模块在处理量化后的 token 时，对它们进行求和归约，并结合多头注意力机制。

    参数:
        codebook_size (int): 码本的大小。
        num_quantizers (int): 量化器的数量。
        conformer (Conformer 或 dict[str, Any]): Conformer 模型或其参数字典。
        grouped_quantizers (int, 可选): 分组量化器的数量。默认值为 1。
    """

    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_quantizers,
        conformer: Conformer | dict[str, Any],
        grouped_quantizers = 1
    ):
        super().__init__()
        # 保存 Conformer 模型
        self.conformer = conformer

        if isinstance(conformer, dict):
            # 如果传入的是参数字典，则实例化 Conformer 模型
            self.conformer = Conformer(**self.conformer)

        # 获取 Conformer 模型的维度
        dim = self.conformer.dim

        # 定义嵌入投影层，如果 grouped_quantizers 大于 1，则进行线性变换和层归一化
        self.embedding_proj = nn.Sequential(
            nn.Linear(dim * grouped_quantizers, dim),
            nn.LayerNorm(dim)
        ) if grouped_quantizers > 1 else nn.Identity()

        # 码本大小加上掩码标记
        num_codes_with_mask = codebook_size + 1
        # 有效量化器的数量
        num_effective_quantizers = num_quantizers * grouped_quantizers

        # 定义码本嵌入层
        self.code_embeds = nn.Embedding(num_codes_with_mask * num_effective_quantizers, dim)

        # 注册量化器偏移量缓冲区
        self.register_buffer('quantizer_offsets', torch.arange(num_effective_quantizers) * num_codes_with_mask, persistent = False)
        # 注册掩码标记缓冲区
        self.register_buffer('mask_tokens', self.quantizer_offsets + num_codes_with_mask, persistent = False)

        # 保存维度
        self.dim = dim
        # 保存码本大小
        self.codebook_size = codebook_size

        # 保存码本大小加上掩码标记
        self.num_codes_with_mask = num_codes_with_mask
        # 保存量化器数量
        self.num_quantizers = num_quantizers
        # 保存分组量化器数量
        self.grouped_quantizers = grouped_quantizers

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * num_effective_quantizers), # 线性变换层，将维度从 dim 增加到 dim * num_effective_quantizers
            Rearrange('b n (h d) -> b (n h) d', h = num_effective_quantizers) # 重塑张量形状
        )
        # 保存有效量化器数量
        self.num_effective_quantizers = num_effective_quantizers

        # each quantizer codebook would require its own logits weight and bias matrices
        # the amazing einops makes this easy with 'EinMix'
        # 定义 logits 映射层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim), # 层归一化
            Rearrange('b (n gq) d -> b n gq d', gq = num_effective_quantizers), # 重塑张量形状
            EinMix(
                'b n gq d -> b n gq l', 
                weight_shape = 'gq d l', # 权重形状
                bias_shape = 'gq l', # 偏置形状
                gq = num_effective_quantizers, # 分组量化器数量
                l = codebook_size, # 码本大小
                d = dim # 维度
            ),
            Rearrange('b ... d -> b (...) d') # 重塑张量形状
        )

    def forward(
        self,
        x,
        *,
        mask = None,
        cond = None,
        sum_embeds = None,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        """
        einops notation:
        b - batch
        n - sequence
        g - groups
        q - quantizers
        d - feature dimension
        """
        """
        前向传播方法，用于处理输入张量并生成输出 logits 或嵌入张量。

        einops 表示法:
        b - batch (批次)
        n - sequence (序列)
        g - groups (分组)
        q - quantizers (量化器)
        d - feature dimension (特征维度)

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length)。
            mask (Tensor, 可选): 掩码张量，用于掩码注意力计算。默认值为 None。
            cond (Tensor, 可选): 条件张量，用于条件信息。默认值为 None。
            sum_embeds (Tensor, 可选): 求和嵌入张量，用于与输入张量相加。默认值为 None。
            return_embeddings (bool, 可选): 是否返回嵌入张量。默认值为 False。
            return_logits_and_embeddings (bool, 可选): 是否同时返回 logits 和嵌入张量。默认值为 False。

        返回:
            Tensor 或 Tuple[Tensor, Tensor]: 如果 return_logits_and_embeddings 为 True，则返回 (logits, embeds)；如果 return_embeddings 为 True，则返回 embeds；否则返回 logits。
        """
        # 获取序列长度、量化器数量和分组数量
        n, q, g = x.shape[-1], self.num_quantizers, self.grouped_quantizers
        # 确保序列长度可以被量化器数量整除
        assert divisible_by(n, g * q), 'sequence must be divisible by number of quantizers'

        # 重塑张量形状
        x = rearrange(x, 'b (n gq) -> b n gq', gq = g * q)
        # 添加量化器偏移量
        x = x + self.quantizer_offsets

        # 应用码本嵌入层
        x = self.code_embeds(x)

        # 对分组量化器进行求和归约
        x = reduce(x, 'b n (g q) d -> b n (g d)', 'sum', g = g)

        # 应用嵌入投影层
        x = self.embedding_proj(x)

        if exists(sum_embeds):
            if sum_embeds.ndim == 3:
                # 如果 sum_embeds 是 3 维张量，则对其进行求和归约
                sum_embeds = reduce(sum_embeds, 'b (n h) d -> b n d', 'sum', h = self.num_effective_quantizers)
            # 将 sum_embeds 加到输入张量上
            x = x + sum_embeds

        if exists(cond):
            if cond.ndim == 2:
                # 如果 cond 是 2 维张量，则重塑其形状
                cond = rearrange(cond, 'b d -> b 1 d')
            # 将条件张量加到输入张量上
            x = x + cond

        # 应用 Conformer 模型
        x = self.conformer(x, mask = mask)
        # 应用多头注意力机制
        embeds = self.heads(x)

        if return_embeddings or not exists(self.to_logits):
            # 如果需要返回嵌入张量或不存在 logits 映射层，则返回嵌入张量
            return embeds

        # 应用 logits 映射层
        logits = self.to_logits(embeds)

        if return_logits_and_embeddings:
            # 如果需要同时返回 logits 和嵌入张量，则返回两者
            return logits, embeds
        # 否则，返回 logits
        return logits


# for main logits as well as self token critic

class LogitHead(Module):
    """
    LogitHead 模块，用于生成主 logits 以及自回归 token critic 的 logits。
    该模块通过一个 ConformerWrapper 模型生成嵌入向量，然后通过一个线性层生成 logits。

    参数:
        net (ConformerWrapper): ConformerWrapper 模型，用于生成嵌入向量。
        logit_dim (int): 输出 logits 的维度。
    """
    def __init__(
        self,
        net: ConformerWrapper,
        logit_dim
    ):
        super().__init__()
        # 保存 ConformerWrapper 模型
        self.net = net
        # 获取 ConformerWrapper 模型的维度
        dim = net.dim
        # 定义线性层，将维度从 dim 映射到 logit_dim
        self.to_logits = nn.Linear(dim, logit_dim)

    def forward(self, x):
        """
        前向传播方法，用于生成 logits。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 生成的 logits。
        """
        # 通过 ConformerWrapper 模型生成嵌入向量
        embed = self.net(x, return_embeddings = True)
        # 通过线性层生成 logits
        return self.to_logits(embed)


# main soundstorm class, which is just a maskgit
# 定义损失分解命名元组，用于存储生成器和 critic 的损失
LossBreakdown = namedtuple('LossBreakdown', ['generator_loss', 'critic_loss'])


class SoundStorm(Module):
    """
    SoundStorm 模型类，结合了 ConformerWrapper 和其他模块，用于音频生成和自回归token。

    参数:
        net (ConformerWrapper): 包装后的 Conformer 模型。
        soundstream (SoundStream, 可选): SoundStream 模型，用于音频处理。默认值为 None。
        spear_tts_text_to_semantic (TextToSemantic, 可选): SpearTTS 的文本到语义模块，用于语义处理。默认值为 None。
        wav2vec (HubertWithKmeans 或 FairseqVQWav2Vec, 可选): wav2vec 模型，用于音频特征提取。默认值为 None。
        steps (int, 可选): 训练步骤的数量。默认值为 18。
        self_cond (bool, 可选): 是否使用自回归条件。默认值为 False。
        self_cond_train_prob (float, 可选): 自回归条件训练的概率。默认值为 0.75。
        no_replace_prob (float, 可选): 在原始 MLM 论文中，保持相同的掩码 token 的概率。默认值为 0.15。
        random_token_prob (float, 可选): 替换为随机 token 的掩码 token 的概率。默认值为 0.1。
        schedule (str 或 callable, 可选): 调度函数，可以是 'linear' 或 'cosine'，也可以是自定义的 callable。默认值为 'linear'。
        can_mask_prev_unmasked (bool, 可选): 在解掩码时，是否可以重新掩码之前未掩码的 token。默认值为 True。
        self_token_critic (bool, 可选): 是否使用自回归 token 批评。默认值为 False。
        critic_loss_weight (float, 可选): 批评损失的权重。默认值为 1.0。
        num_semantic_token_ids (int, 可选): 语义 token ID 的数量。如果使用语义条件，则必须提供。默认值为 None。
        semantic_pad_id (int, 可选): 语义填充 ID。默认值为 -1。
        pad_id (int, 可选): 填充 ID。默认值为 None。
        wav2vec_target_sample_hz (int, 可选): wav2vec 模型的目标采样率。默认值为 None。
        wav2vec_downsample_factor (int, 可选): wav2vec 模型的降采样因子。默认值为 None。
        codec_target_sample_hz (int, 可选): 编码器的目标采样率。默认值为 None。
        codec_downsample_factor (int, 可选): 编码器的降采样因子。默认值为 None。
    """

    @beartype
    def __init__(
        self,
        net: ConformerWrapper,
        *,
        soundstream: SoundStream | None = None,
        spear_tts_text_to_semantic: TextToSemantic | None = None,
        wav2vec: HubertWithKmeans | FairseqVQWav2Vec | None = None,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        no_replace_prob = 0.15,          # which percentage of the tokens masked will stay the same, done in original MLM paper
        random_token_prob = 0.1,         # which percentage of tokens to be replaced with random token, done in original MLM paper
        schedule = 'linear',
        can_mask_prev_unmasked = True,   # when unmasking, whether it can remask previously unmasked
        self_token_critic = False,       # https://aclanthology.org/2021.naacl-main.409/
        critic_loss_weight = 1.,
        num_semantic_token_ids = None,
        semantic_pad_id = -1,
        pad_id = None,
        wav2vec_target_sample_hz = None,
        wav2vec_downsample_factor = None,
        codec_target_sample_hz = None,
        codec_downsample_factor = None,
    ):
        super().__init__()

        # conformer settings

        # 保存 ConformerWrapper 模型
        self.net = net
        # 获取 ConformerWrapper 模型的维度
        dim = net.dim
        # 保存维度
        self.dim = dim
        # 获取码本大小
        self.num_tokens = net.codebook_size
        # 保存填充 ID
        self.pad_id = pad_id

        # set soundstream

        # 保存 SoundStream 模型
        self.soundstream = soundstream

        if exists(soundstream):
            # 获取编码器的目标采样率
            self.codec_target_sample_hz = soundstream.target_sample_hz
            # 获取编码器的降采样因子
            self.codec_downsample_factor = soundstream.downsample_factor
        else:
            # 如果没有 SoundStream，则使用提供的编码器目标采样率
            self.codec_target_sample_hz = codec_target_sample_hz
            # 如果没有 SoundStream，则使用提供的编码器降采样因子
            self.codec_downsample_factor = codec_downsample_factor

        if exists(self.soundstream):
            # 确保 ConformerWrapper 的分组量化器与 SoundStream 的 rq_groups 一致
            assert net.grouped_quantizers == soundstream.rq_groups
            # 确保 ConformerWrapper 的码本大小与 SoundStream 的 codebook_size 一致
            assert net.codebook_size == soundstream.codebook_size
            # 确保 ConformerWrapper 的量化器数量与 SoundStream 的 num_quantizers 一致
            assert net.num_quantizers == soundstream.num_quantizers

        # set text-to-semantic

        # 保存 SpearTTS 的文本到语义模块
        self.text_to_semantic = spear_tts_text_to_semantic

        if exists(spear_tts_text_to_semantic) and exists(spear_tts_text_to_semantic.wav2vec):
            # 如果 SpearTTS 提供了 wav2vec，则不需要再提供
            assert not exists(wav2vec), 'wav2vec model already supplied from the TextToSemantic instance from SpearTTS'
            # 确保没有提供 wav2vec 的降采样因子和采样率
            assert not (exists(wav2vec_downsample_factor) or exists(wav2vec_target_sample_hz)), 'wav2vec downsample factor and sampling freq being auto-set from the text-to-semantic module passed in, as it contains the wav2vec instance'

            # 设置 wav2vec 模型
            self.wav2vec = spear_tts_text_to_semantic.wav2vec
            # 获取 wav2vec 的目标采样率
            self.wav2vec_target_sample_hz = maybe_wav2vec.target_sample_hz
            # 获取 wav2vec 的降采样因子
            self.wav2vec_downsample_factor = maybe_wav2vec.downsample_factor

        elif exists(wav2vec):
            # 确保没有提供 wav2vec 的降采样因子和采样率
            assert not (exists(wav2vec_downsample_factor) or exists(wav2vec_target_sample_hz)), 'wav2vec downsample factor and sampling freq being auto-set from the text-to-semantic module passed in, as it contains the wav2vec instance'

            # 设置 wav2vec 模型
            self.wav2vec = wav2vec
            # 获取 wav2vec 的目标采样率
            self.wav2vec_target_sample_hz = wav2vec.target_sample_hz
            # 获取 wav2vec 的降采样因子
            self.wav2vec_downsample_factor = wav2vec.downsample_factor

        else:
            # 如果没有提供 wav2vec，则设置为 None
            self.wav2vec = None
            # 使用提供的 wav2vec 目标采样率
            self.wav2vec_target_sample_hz = wav2vec_target_sample_hz
            # 使用提供的 wav2vec 降采样因子
            self.wav2vec_downsample_factor = wav2vec_downsample_factor

        # whether to text condition on audio generation is dependent on whether hyperparameters are supplied
        # 是否对音频生成进行文本条件化取决于是否提供了超参数
        self.should_condition = exists(self.wav2vec_downsample_factor) and exists(self.wav2vec_target_sample_hz)

        # in the case that text-to-semantic module passed in
        # 在传递了文本到语义模块的情况下

        if self.should_condition:
            # 确保提供了编码器的目标采样率和降采样因子
            assert exists(self.codec_target_sample_hz) and exists(self.codec_downsample_factor)

            if exists(spear_tts_text_to_semantic):
                # 设置语义 token 嵌入
                self.semantic_token_emb = spear_tts_text_to_semantic.semantic_token_emb
                # 获取语义 token ID 的数量
                self.num_semantic_token_ids = spear_tts_text_to_semantic.num_semantic_token_ids
                # 定义线性变换层，将语义模型维度映射到 ConformerWrapper 模型维度
                self.semantic_cond_to_model_dim = nn.Linear(spear_tts_text_to_semantic.dim, net.dim)
                # 获取语义填充 ID
                self.semantic_pad_id = spear_tts_text_to_semantic.pad_id.get('speech')
            else:
                # 如果没有提供 SpearTTS，则需要提供语义 token ID 的数量
                assert exists(num_semantic_token_ids), 'if you are conditioning, you must pass in the number of semantic token ids'
                # 定义语义 token 嵌入层
                self.semantic_token_emb = nn.Embedding(num_semantic_token_ids, dim)
                # 保存语义 token ID 的数量
                self.num_semantic_token_ids = num_semantic_token_ids
                # 定义恒等变换层
                self.semantic_cond_to_model_dim = nn.Identity()
                # 设置语义填充 ID
                self.semantic_pad_id = semantic_pad_id

        # detect token critic settings
        # 获取量化器的数量
        self.num_quantizers = net.num_quantizers
        # 获取分组量化器的数量
        self.grouped_quantizers = net.grouped_quantizers

        # 设置掩码 ID
        self.mask_id = net.codebook_size

        # afaict, maskgit paper did not do this
        # but may help for self conditioning, as used successfully in original BERT
        # AFAICT，MaskGIT 论文中没有这样做
        # 但可能有助于自条件化，正如在原始的 BERT 中成功使用的那样

        # 设置保持相同的掩码 token 的概率
        self.no_replace_prob = no_replace_prob
        # 设置替换为随机 token 的掩码 token 的概率
        self.random_token_prob = random_token_prob

        # 设置训练步骤的数量
        self.steps = steps

        if callable(schedule):
            # 如果调度函数是 callable，则使用它
            self.schedule_fn = schedule
        if schedule == 'linear':
            # 如果调度函数是 'linear'，则使用线性调度
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            # 如果调度函数是 'cosine'，则使用余弦调度
            self.schedule_fn = cosine_schedule
        else:
            # 如果调度函数无效，则抛出异常
            raise ValueError(f'invalid schedule {schedule}')

        # 设置是否可以在解掩码时重新掩码之前未掩码的 token
        self.can_mask_prev_unmasked = can_mask_prev_unmasked

        # self conditioning
        # 设置自条件化标志
        self.self_cond = self_cond

        if self_cond:
            # 定义 null 嵌入
            self.null_embed = nn.Parameter(torch.randn(dim))
            # 定义线性变换层，用于自条件化
            self.to_self_cond = nn.Linear(dim, dim, bias = False) if self_cond else None
            # 设置自条件化训练概率
            self.self_cond_train_prob = self_cond_train_prob

        # token critic

        self.token_critic = None
        if self_token_critic:
            # 如果启用 token critic，则定义 LogitHead
            self.token_critic = LogitHead(net, 1)

        # 设置critic损失的权重
        self.critic_loss_weight = critic_loss_weight

    @property
    def device(self):
        return next(self.net.parameters()).device

    def load(self, path, strict = True):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        num_latents = None,
        *,
        mask = None,
        texts: list[str] | Tensor | None = None,
        cond_semantic_token_ids = None,
        prompt_acoustic_token_ids = None,
        seconds = None,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.7,
        noise_level_scale = 1.,
        num_full_sampling_levels = 1,
        text_to_semantic_generate_kwargs: dict = {},
        spec_decode = False,
        spec_decode_gamma = 5,
        **kwargs
    ):
        """
        生成音频数据。

        参数:
            num_latents (int, 可选): 潜在向量的数量。
            mask (Tensor, 可选): 掩码张量。
            texts (list[str] 或 Tensor, 可选): 输入文本。
            cond_semantic_token_ids (Tensor, 可选): 条件语义 token ID。
            prompt_acoustic_token_ids (Tensor, 可选): 提示声学 token ID。
            seconds (float, 可选): 生成音频的时长。
            batch_size (int, 可选): 批大小。
            start_temperature (float, 可选): 起始温度。默认值为 1.0。
            filter_thres (float, 可选): 过滤阈值。默认值为 0.7。
            noise_level_scale (float, 可选): 噪声水平缩放因子。默认值为 1.0。
            num_full_sampling_levels (int, 可选): 完全采样级别的数量。默认值为 1。
            text_to_semantic_generate_kwargs (dict, 可选): 文本到语义生成的附加关键字参数。
            spec_decode (bool, 可选): 是否进行频谱解码。默认值为 False。
            spec_decode_gamma (float, 可选): 频谱解码的 gamma 参数。默认值为 5。
            **kwargs: 其他关键字参数。

        返回:
            Tensor: 生成音频数据的张量。
        """
        if self.should_condition and not exists(cond_semantic_token_ids):
            assert exists(texts) and exists(self.text_to_semantic)

            if is_bearable(texts, List[str]):
                assert exists(self.text_to_semantic.tokenizer_encode)
                # 对文本进行编码
                texts = self.text_to_semantic.tokenizer_encode(texts)
                # 将文本张量移动到设备
                texts = texts.to(self.device)

            # 生成条件语义 token ID
            cond_semantic_token_ids = self.text_to_semantic.generate(
                texts,
                source_type = 'text',
                target_type = 'speech',
                spec_decode = spec_decode,
                spec_decode_gamma = spec_decode_gamma,
                **text_to_semantic_generate_kwargs
            )

        assert not (exists(cond_semantic_token_ids) ^ self.should_condition), 'you either have text-conditioning turned on and have not passed in any conditioning semantic token ids, or vice versa'

        # maybe condition
        # 条件化处理
        # 获取条件化 token
        cond_tokens = self.maybe_get_condition(cond_semantic_token_ids)

        # determine batch size and sequence length, which depends whether it is conditioning
        # 确定批大小和序列长度，这取决于是否进行条件化
        if exists(cond_tokens):
            # 获取批大小和潜在向量数量
            batch_size, num_latents = cond_tokens.shape[:2]
            # 判断是否为单样本
            sample_one = batch_size == 1
        else:
            sample_one = not exists(batch_size)
            batch_size = default(batch_size, 1)

            assert exists(num_latents) ^ exists(seconds)

            if not exists(num_latents):
                assert exists(self.soundstream), 'soundstream must be passed in to generate in seconds'
                # 计算潜在向量数量
                num_latents = (seconds * self.soundstream.target_sample_hz) // self.soundstream.seq_len_multiple_of

        # determine sequence length
        # 确定序列长度
        # 计算有效的量化器数量
        num_effective_quantizers = self.grouped_quantizers * self.num_quantizers
        # 计算序列长度
        seq_len = num_latents * num_effective_quantizers

        # device and time

        device = self.device

        # 生成时间步长
        times = torch.linspace(0., 1., self.steps + 1, device = device)

        # sequence starts off as all masked
        # todo: find a better name for sequence mask vs mask for mask diffusion
        # 序列开始时全部被掩码

        # 定义形状
        shape = (batch_size, seq_len)

        # 初始化序列为全掩码
        seq = torch.full(shape, self.mask_id, device = device)

        # 设置掩码
        seq_mask = mask

        if not exists(seq_mask):
            # 如果没有提供掩码，则初始化为全 1
            seq_mask = torch.ones((batch_size, num_latents), device = device, dtype = torch.bool)

        # 重复掩码以匹配量化器数量
        seq_mask_with_quantizer = repeat(seq_mask, 'b n -> b (n q)', q = num_effective_quantizers)

        # 初始化掩码为全 True
        mask = torch.full(shape, True, device = device)
        
        # include prompt tokens unmasked as the sequence prefix, starting from the lowest quantizer
        # 将提示 token 作为序列前缀，保持未掩码，从最低量化器开始
        # 初始化提示掩码
        prompt_mask = None
        
        if exists(prompt_acoustic_token_ids):
            # 获取提示声学 token ID 的长度和量化器数量
            prompt_len, num_prompt_quantizers = prompt_acoustic_token_ids.shape[1:]
            assert num_prompt_quantizers <= num_effective_quantizers, 'number of prompt quantizers cannot be greater than the number of quantizers'
            
            # 重塑序列形状
            seq = rearrange(seq, 'b (n q) -> b n q', q = num_effective_quantizers)
            # 重塑掩码形状
            prompt_mask = rearrange(mask, 'b (n q) -> b n q', q = num_effective_quantizers)
            
            # 将提示声学 token ID 填充到序列中
            seq[:, :prompt_len, :num_prompt_quantizers] = prompt_acoustic_token_ids
            # 设置提示掩码为 False
            prompt_mask[:, :prompt_len, :num_prompt_quantizers] = False
            
            # 重塑回原始形状
            seq = rearrange(seq, 'b n q -> b (n q)', q = num_effective_quantizers)
            # 重塑回原始形状
            prompt_mask = rearrange(prompt_mask, 'b n q -> b (n q)', q = num_effective_quantizers)

        # slowly demask
        # 逐步解掩码
        # 计算序列长度
        seq_len_from_mask = reduce(seq_mask, 'b n -> b', 'sum')

        # 获取随机掩码概率
        rand_mask_probs = self.schedule_fn(times[1:])
        # 重塑随机掩码概率形状
        rand_mask_probs = rearrange(rand_mask_probs, 'n -> n 1')

        # 计算所有掩码 token 的数量
        all_mask_num_tokens = (rand_mask_probs * seq_len_from_mask).long()

        prev_mask = None

        # self conditioning
        # 判断是否使用自回归条件
        has_self_cond = self.self_cond
        # 如果使用自回归条件，则使用 null_embed 作为上一次的嵌入向量
        last_embed = self.null_embed if has_self_cond else None
        
        # 遍历每一个量化器层级
        for q_level in range(num_effective_quantizers):
            # 确定当前量化器层级需要掩码的 token 数量
            mask_num_tokens_for_q_level = all_mask_num_tokens if q_level < num_full_sampling_levels else torch.zeros((1, batch_size), dtype = torch.long, device = device)

            for mask_num_tokens, steps_until_x0 in tqdm(zip(mask_num_tokens_for_q_level, reversed(range(self.steps))), total = self.steps):
                # 如果使用自回归条件，则计算自回归条件嵌入向量
                self_cond = self.to_self_cond(last_embed) if has_self_cond else None

                # 通过 ConformerWrapper 模型生成 logits 和嵌入向量
                logits, embeds = self.net(
                    seq,
                    mask = seq_mask,
                    cond = cond_tokens,
                    sum_embeds = self_cond,
                    return_logits_and_embeddings = True,
                    **kwargs
                )

                if has_self_cond:
                    # 更新上一次的嵌入向量
                    last_embed = embeds

                if exists(filter_thres):
                    # 对 logits 应用 top-k 过滤
                    logits = top_k(logits, filter_thres)

                # 计算annealing比例和temperature
                annealing_scale = steps_until_x0 / self.steps
                temperature = start_temperature * annealing_scale

                # 使用 Gumbel 采样生成采样 ID
                sampled_ids = gumbel_sample(logits, temperature = max(temperature, 1e-3))

                # don't sample for lower quantizer levels
                # 对于高于 0 的量化器层级，不进行采样
                if q_level > 0:
                    sample_mask = rearrange(mask, 'b (n q) -> b n q', q = num_effective_quantizers)
                    sample_mask[:, :, :q_level] = False
                    sample_mask = rearrange(sample_mask, 'b n q -> b (n q)', q = num_effective_quantizers)
                else:
                    # 如果是第一个量化器层级，则使用原始掩码
                    sample_mask = mask

                # 根据掩码更新序列
                seq = torch.where(sample_mask, sampled_ids, seq)
                
                if (mask_num_tokens == 0).all():
                    # 如果没有需要掩码的 token，则跳过当前循环
                    continue

                if exists(self.token_critic):
                    scores = self.token_critic(seq) # 使用 token 批评模型生成评分
                    scores = rearrange(scores, 'b n 1 -> b n') # 重塑评分张量形状
                    scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale # 添加噪声
                else:
                    scores = 1 - logits.softmax(dim = -1) # 计算评分（不使用 token 批评模型）
                    scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                    scores = rearrange(scores, 'b n 1 -> b n')

                # 初始化掩码
                mask = torch.zeros_like(scores, dtype = torch.bool)
                
                # mask based on highest score
                # 基于最高评分进行掩码
                # 计算掩码值
                mask_value = -torch.finfo(scores.dtype).max

                # 掩码掉不需要掩码的位置
                scores = scores.masked_fill(~seq_mask_with_quantizer, mask_value)

                if not self.can_mask_prev_unmasked and exists(prev_mask):
                    # 如果不允许重新掩码之前未掩码的 token，则掩码掉这些位置
                    scores = scores.masked_fill(~prev_mask, mask_value)

                # 对评分进行排序
                scores_sorted = scores.argsort(dim = -1, descending = True).argsort(dim = -1)

                # 重塑掩码数量张量形状
                mask_num_tokens = rearrange(mask_num_tokens, 'b -> b 1')

                # 生成掩码
                mask = scores_sorted < mask_num_tokens

                if not self.can_mask_prev_unmasked:
                    # 记录当前掩码状态
                    prev_mask = mask.clone()

                # 重塑掩码张量形状
                mask = rearrange(mask, 'b (n q) -> b n q', q = num_effective_quantizers)
                
                # mask all upper quantizer levels
                # 掩码所有更高的量化器层级
                if q_level < (num_effective_quantizers - 1):
                    mask[:, :, q_level + 1:] = True
                    
                # unmask all lower quantizer levels
                # 取消掩码所有更低的量化器层级
                if q_level > 0:
                    mask[:, :, :q_level] = False

                # 重塑回原始形状
                mask = rearrange(mask, 'b n q -> b (n q)', q = num_effective_quantizers)

                if exists(prompt_mask):
                    mask = mask & prompt_mask
                
                # 应用掩码到序列中
                seq = seq.masked_fill(mask, self.mask_id)

        out = seq

        if exists(self.soundstream):
            # 重塑序列形状以匹配 SoundStream 的输入
            seq = rearrange(seq, 'b (n q) -> b n q', q = self.num_quantizers)

            with torch.no_grad():
                self.soundstream.eval()
                out = self.soundstream.decode_from_codebook_indices(seq) # 解码序列
                out = rearrange(out, 'b 1 ... -> b ...') # 重塑输出张量形状

        if sample_one:
            # 如果是单样本，则重塑输出张量形状
            out = rearrange(out, '1 ... -> ...')

        return out

    def maybe_get_condition(self, token_ids = None, length = None):
        """
        根据给定的语义 token ID 获取条件化 token。

        参数:
            token_ids (Tensor, 可选): 语义 token ID 张量。
            length (int, 可选): 目标条件化 token 的长度。

        返回:
            Tensor 或 None: 条件化 token 张量或 None。
        """
        # 确保只有在需要条件化时才传递 token_ids
        assert not (exists(token_ids) ^ self.should_condition), 'you either have text-conditioning turned on and have not passed in any conditioning semantic token ids, or vice versa'

        if not exists(token_ids):
            # 如果没有提供 token_ids，则返回 None
            return None
        
        # 根据是否存在 text_to_semantic 模块选择上下文管理器
        context = torch.no_grad if exists(self.text_to_semantic) else nullcontext

        with context():
            # 创建掩码，标记非填充的 token
            mask = token_ids != self.semantic_pad_id

            # also remove the eos semantic token id
            # 如果存在 text_to_semantic 模块并且设置了 speech 的 eos_id，则进一步过滤掉 eos token

            if exists(self.text_to_semantic) and self.text_to_semantic.autoset_eos_id['speech']:
                mask &= token_ids != self.num_semantic_token_ids

            # 将非掩码的 token 填充为 0
            token_ids = token_ids.masked_fill(~mask, 0)

            # 通过语义 token 嵌入层获取语义 tokens
            semantic_tokens = self.semantic_token_emb(token_ids)
            # 将语义 tokens 转换为模型维度的条件化 tokens
            cond_tokens = self.semantic_cond_to_model_dim(semantic_tokens)

            # just mask out the padding to 0s and let the network learn that for now
            # eventually should add self attention masking to conformer, and calculate the correct number of masked tokens per variable lengthed batch row
            # 暂时仅将填充部分掩码为 0，并让网络学习
            # 最终应该添加自注意力掩码到 conformer，并计算每个可变长度批次行的正确掩码 token 数量

            cond_tokens = cond_tokens.masked_fill(~rearrange(mask, '... -> ... 1'), 0.)

        # now need to interpolate the conditioning tokens
        # to align semantic and vector quantized tokens, time-wise
        # 现在需要插值条件化 tokens
        # 以便在时间上对齐语义和向量量化 tokens

        # 获取条件化 tokens 的长度
        cond_length = cond_tokens.shape[-2]

        # 计算目标条件化长度
        target_cond_length = math.ceil(cond_length * (self.wav2vec_downsample_factor / self.wav2vec_target_sample_hz) / (self.codec_downsample_factor / self.codec_target_sample_hz))

        # pytorch does not interpolate 1d, so hack by convert to 2d

        if cond_length != target_cond_length:
            cond_tokens = rearrange(cond_tokens, 'b n d -> b d n 1') # 重塑张量形状
            cond_tokens = F.interpolate(cond_tokens, (target_cond_length, 1), mode = 'bilinear') # 进行双线性插值
            cond_tokens = rearrange(cond_tokens, 'b d n 1 -> b n d') # 重塑回原始形状

        # whether to curtail or pad to length
        # 根据目标长度进行裁剪或填充

        # 更新条件化 tokens 的长度
        cond_length = cond_tokens.shape[-2]

        if exists(length):
            if cond_length < length:
                cond_tokens = F.pad(cond_tokens, (0, 0, 0, length - cond_length), value = 0.) # 填充条件化 tokens
            elif cond_length > length:
                cond_tokens = cond_tokens[:, :length] # 裁剪条件化 tokens

        return cond_tokens

    def forward(
        self, 
        x, # 输入张量
        *,
        mask = None, # 掩码张量（可选）
        cond_semantic_token_ids = None, # 条件语义 token ID（可选）
        only_train_generator = False, # 是否仅训练生成器
        only_train_critic = False, # 是否仅训练批评器
        generator_sample_temperature = None, # 生成器采样温度（可选）
        **kwargs
    ):
        # if raw audio passed in, convert to residual quantized vectors
        # 如果传入的是原始音频，则将其转换为残差量化向量

        # 判断输入是否为原始音频
        is_raw_audio = x.dtype == torch.float

        # if semantic token ids not supplied and conditioning is indicated
        # see if wav2vec and raw audio is available
        # 如果没有提供语义 token ID 且需要条件化
        # 检查是否提供了 wav2vec 模型和原始音频

        if self.should_condition and not exists(cond_semantic_token_ids) and is_raw_audio:
            with torch.no_grad():
                self.wav2vec.eval()
                # 使用 wav2vec 生成条件语义 token ID
                cond_semantic_token_ids = self.wav2vec(x, flatten = False)

        # derive residual vector quantized ids if raw audio passed in
        # 如果传入的是原始音频，则推导残差向量量化 ID

        if is_raw_audio:
            # 确保提供了 soundstream 模型
            assert exists(self.soundstream)
            with torch.no_grad():
                self.soundstream.eval()
                # 使用 soundstream 对原始音频进行编码
                _, x, _ = self.soundstream(x, return_encoded = True)

        # shape
        # 获取形状信息
        # 获取批大小、序列长度、量化器数量和设备信息
        b, n, gq, device = *x.shape, x.device

        assert gq == (self.num_quantizers * self.grouped_quantizers), f'codes passed in has {gq} quantizers (x groups) but the conformer wrapper was set to num_quantizers {self.num_quantizers} and grouped_quantizers {self.grouped_quantizers}'

        # mask was used below, rename input mask as seq_mask
        # todo: rename mask used for mask diffusion later
        # 将输入掩码重命名为 seq_mask，后续用于掩码扩散
        # todo: 之后重命名用于掩码扩散的掩码

        # 重命名输入掩码为 seq_mask
        seq_mask = mask

        if not exists(seq_mask):
            # 如果没有提供掩码，则初始化为全 1
            seq_mask = torch.ones((b, n), device = device, dtype = torch.bool)

        if exists(self.pad_id):
            pad_mask = (x == self.pad_id).any(dim = -1) # 创建填充掩码
            seq_mask = seq_mask & ~pad_mask # 更新序列掩码，排除填充部分

            if self.pad_id < 0:
                # if using say -1 for padding
                # 如果使用 -1 作为填充 ID，则将填充部分填充为 0
                x = torch.where(rearrange(pad_mask, 'b n -> b n 1'), 0, x)

        # maybe condition
        # 获取条件化 tokens
        cond_tokens = self.maybe_get_condition(cond_semantic_token_ids, length = x.shape[-2])

        # prepare masking, selecting the prompt from a random prefix
        # 重塑原始序列形状
        orig_seq = rearrange(x.clone(), 'b n q -> b (n q)')
        # 计算最小序列长度
        min_seq_len = seq_mask.sum(dim = -1).amin()
        # 随机选择一个时间步长
        t = randrange(0, min_seq_len - 1)

        # 生成掩码，从时间步长 t 开始
        mask = seq_mask[:, t:]

        # 生成掩码，从时间步长 t 开始
        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        # 计算随机掩码概率
        rand_probs = self.schedule_fn(rand_times)

        # 获取掩码子集
        mask = get_mask_subset_prob(mask, rand_probs, min_mask = 1)

        # random quantizer position, in groups
        # 随机量化器位置，按组计算
        # 随机选择一个量化器位置
        q = randrange(0, self.num_quantizers) * self.grouped_quantizers

        # to ensure all tokens produce embeddings, instead of just the ones with [mask] input, as done in seminal BERT MLM paper
        # potentially needed for self-conditioning (on embedding) to work well
        # 为了确保所有 tokens 都产生嵌入向量，而不仅仅是输入为 [mask] 的 tokens，如原始 BERT MLM 论文中所做的那样
        # 可能需要自条件化（基于嵌入向量）才能良好工作

        # 复制掩码
        replace_mask_id_mask = mask.clone()
        # 初始化剩余序列比例
        frac_seq_left = 1.

        # 如果 no_replace_prob 大于 0 且随机选择，则不替换部分 tokens
        if self.no_replace_prob > 0. and coin_flip():
            # 更新剩余序列比例
            frac_seq_left -= self.no_replace_prob

            # 获取不替换掩码
            no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
            # 更新替换掩码
            replace_mask_id_mask &= ~no_replace_prob_mask

        # 如果 random_token_prob 大于 0 且随机选择，则替换部分 tokens 为随机 tokens
        if self.random_token_prob > 0. and coin_flip():
            # 获取随机替换掩码
            random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left, min_keep_mask = 1)
            # 生成随机 tokens
            random_tokens = torch.randint(0, self.num_tokens, (b, n - t), device = device)

            # 替换 tokens
            x[:, t:, q] = torch.where(random_token_prob_mask, random_tokens, x[:, t:, q])
            # 更新替换掩码
            replace_mask_id_mask &= ~random_token_prob_mask

        # 替换 [mask] token
        # 根据 replace_mask_id_mask 替换为 [mask] token
        masked = torch.where(replace_mask_id_mask, self.mask_id, x[:, t:, q])
        # 将替换后的部分与前面的部分拼接，并重塑形状
        masked = rearrange(torch.cat((x[:, :t, q], masked), dim=1), 'b n -> b n 1')
        # 将拼接后的部分与其他量化器组拼接
        masked = torch.cat((x[:, :, :q], masked, x[:, :, q + 1:]), dim=2)
        # 将后续量化器组的 token 替换为 [mask]
        masked[:, t:, q + 1:] = self.mask_id
        # 重塑回原始形状
        masked = rearrange(masked, 'b n q -> b (n q)')

        # 初始化提示掩码为全 False
        prompt_mask = torch.full((b, t), False, device=device)
        # 初始化低量化器掩码为全 False
        lower_quantizers_mask = torch.full((b, n, q), False, device=device)
        # 初始化高量化器掩码为全 True
        upper_quantizers_mask = torch.full((b, n, (gq - q - 1)), True, device=device)

        # upper_quantizers_mask in prompt also should be False
        # 在提示部分，高量化器掩码设为 False
        upper_quantizers_mask[:, :t, :] = False
        # 将提示掩码和替换掩码拼接，并重塑形状
        mask = rearrange(torch.cat((prompt_mask, replace_mask_id_mask), dim=1), 'b n -> b n 1')
        # 将低量化器掩码、拼接后的掩码和高量化器掩码拼接
        mask = torch.cat((lower_quantizers_mask, mask, upper_quantizers_mask), dim = 2)

        # above is the right mask, but when computing loss, only consider level q
        # 上述是计算损失时的正确掩码，但在计算损失时，只考虑量化器级别 q
        # 将高量化器级别的掩码设为 False
        mask[:, :, q + 1:] = False
        # 重塑回原始形状
        mask = rearrange(mask, 'b n q -> b (n q)')

        # self conditioning

        if self.self_cond:
            # 初始化自条件化嵌入向量
            self_cond = self.null_embed

            if sample_prob(self.self_cond_train_prob):
                with torch.no_grad():
                    self_cond = self.net(
                        masked,
                        cond = cond_tokens,
                        return_embeddings = True,
                        mask = seq_mask,
                        **kwargs
                    ).detach() # 使用网络生成自条件化嵌入向量，并将其分离

            # 更新关键字参数，加入自条件化嵌入向量
            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        # logits
        # 如果只训练批评器，则使用无梯度上下文；否则，使用空上下文
        context = torch.no_grad if only_train_critic else nullcontext

        with context():
            logits = self.net(
                masked,
                mask = seq_mask,
                cond = cond_tokens,
                **kwargs
            )  # 使用网络生成 logits

        # cross entropy loss
        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits[mask],
            orig_seq[mask]
        )

        if not exists(self.token_critic) or only_train_generator:
            # 如果不存在 token 批评模型或只训练生成器，则返回损失和损失分解
            return loss, LossBreakdown(loss, None)

        # 生成采样 ID
        # 使用 Gumbel 采样生成采样 ID
        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        # 根据掩码生成生成的序列
        generated = torch.where(mask, sampled_ids, orig_seq)

        # 计算批评损失
        # 使用 token 批评模型生成批评 logits
        critic_logits = self.token_critic(generated)
        # 生成批评标签
        critic_labels = (sampled_ids != orig_seq).float()

        # 计算二元交叉熵损失
        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'), # 重塑批评 logits 形状
            critic_labels
        )

        # determine losses to be returned based on what researcher wants to train
        # 根据研究人员的训练需求确定返回的损失
        if only_train_critic:
            # 如果只训练批评器，则总损失为批评损失
            total_loss = critic_loss  
            # 生成器损失为 None
            loss = None 
        else:
            # 否则，总损失为生成器损失加上批评损失乘以批评损失权重
            total_loss = loss + critic_loss * self.critic_loss_weight

        # 返回总损失和损失分解
        return total_loss, LossBreakdown(loss,  critic_loss)
