from __future__ import annotations

from pathlib import Path
import re
from shutil import rmtree

from beartype import beartype
from accelerate import Accelerator, DistributedType

import torch
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, random_split

from audiolm_pytorch.data import get_dataloader
from audiolm_pytorch.optimizer import get_optimizer

from soundstorm import SoundStorm


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


def noop(*args, **kwargs):
    pass


def cycle(dl):
    """
    创建一个无限循环的数据加载器生成器。

    参数:
        dl: 需要循环的数据加载器。

    返回:
        generator: 一个生成器，可以无限循环地生成数据。
    """
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    """
    将输入转换为元组，如果输入已经是元组或列表，则保持不变。

    参数:
        t: 需要转换的输入。

    返回:
        tuple: 转换后的元组。
    """
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    """
    提示用户输入 yes 或 no，并返回布尔值。

    参数:
        question (str): 提示用户的问题。

    返回:
        bool: 如果用户输入 'yes' 或 'y'，则返回 True；否则返回 False。
    """
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    """
    将新的日志累加到现有的日志中。

    参数:
        log (dict): 现有的日志字典。
        new_logs (dict): 新的日志字典。

    返回:
        dict: 更新后的日志字典。
    """
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    """
    根据检查点文件的文件名返回训练的步数。

    文件名格式假设为类似 "/path/to/soundstorm.20000.pt" 的形式，
    表示训练了 20k 步。在这种情况下，返回 20000。

    参数:
        checkpoint_path (str): 检查点文件的路径。

    返回:
        int: 训练的步数。如果文件名中未找到数字，则返回 0。
    """
    # 使用正则表达式查找文件名中的所有数字
    results = re.findall(r'\d+', str(checkpoint_path))
    """
    re.findall(r'\d+', str(checkpoint_path)):
        - r'\d+' 是一个正则表达式，匹配一个或多个连续的数字。
        - re.findall 会返回列表，包含所有匹配的数字字符串。
        - str(checkpoint_path) 将路径转换为字符串，以防路径对象不是字符串类型。
    """

    if len(results) == 0:
        return 0
    
    # 返回列表中的最后一个数字，假设这是训练步数
    return int(results[-1])


class SoundStormTrainer(Module):
    """
    SoundStorm 训练器类，用于训练 SoundStorm 模型。

    参数:
        model (SoundStorm): 需要训练的 SoundStorm 模型。
        num_train_steps (int): 总的训练步数。
        num_warmup_steps (int): 预热的步数。
        batch_size (int): 每个批次的样本数量。
        dataset (Dataset, 可选): 训练数据集。默认值为 None。
        only_train_generator (bool, 可选): 是否仅训练生成器。默认值为 False。
        only_train_critic (bool, 可选): 是否仅训练批评器。默认值为 False。
        lr (float, 可选): 学习率。默认值为 3e-4。
        initial_lr (float, 可选): 初始学习率。默认值为 1e-5。
        grad_accum_every (int, 可选): 梯度累积的步数。默认值为 1。
        wd (float, 可选): 权重衰减。默认值为 0.0。
        max_grad_norm (float, 可选): 梯度裁剪的最大范数。默认值为 0.5。
        valid_frac (float, 可选): 验证集的比例。默认值为 0.05。
        random_split_seed (int, 可选): 随机分割数据集的随机种子。默认值为 42。
        save_results_every (int, 可选): 保存结果的频率（步数）。默认值为 100。
        save_model_every (int, 可选): 保存模型的频率（步数）。默认值为 1000。
        results_folder (str, 可选): 保存结果的文件夹路径。默认值为 './results'。
        accelerate_kwargs (dict, 可选): 传递给 Accelerator 的其他关键字参数。默认值为空字典。
        split_batches (bool, 可选): 是否在多个设备之间分割批次。默认值为 False。
        drop_last (bool, 可选): 是否丢弃最后一个不完整的批次。默认值为 False。
        force_clear_prev_results (bool, 可选): 是否强制清除之前的实验结果。默认值为 None。
    """
    @beartype
    def __init__(
        self,
        model: SoundStorm,
        *,
        num_train_steps,
        num_warmup_steps,
        batch_size,
        dataset: Dataset | None = None,
        only_train_generator = False,
        only_train_critic = False,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None
    ):
        super().__init__()

        # 初始化 Accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            **accelerate_kwargs
        )

        # 保存模型
        self.model = model

        # 注册步数缓冲区
        self.register_buffer('steps', torch.Tensor([0]))

        # 保存训练参数
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        
        self.only_train_generator = only_train_generator
        self.only_train_critic = only_train_critic

        # optimizer
        # 初始化优化器
        self.optim = get_optimizer(
            model.parameters(),
            lr = lr,
            wd = wd
        )

        self.lr = lr
        self.initial_lr = initial_lr
        self.scheduler = CosineAnnealingLR(self.optim, T_max = num_train_steps)

        # max grad norm
        # 设置最大梯度范数
        self.max_grad_norm = max_grad_norm

        # create dataset
        # 创建数据集
        self.ds = dataset

        # split for validation
        # 分割验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader
        # 创建数据加载器
        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # prepare with accelerator
        # 使用 Accelerator 准备模型、优化器、学习率调度器以及数据加载器
        (
            self.model,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.model,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators
        # 创建数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        # 如果是主进程，并且 force_clear_prev_results 为 True 或者没有设置 force_clear_prev_results 且结果文件夹不为空，则询问是否清除之前的实验结果
        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        # 初始化追踪器
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": num_warmup_steps, "learning_rate": lr, "initial_learning_rate": lr}
        self.accelerator.init_trackers("soundstorm", config=hps)

    def save(self, path):
        """
        保存模型、优化器和学习率调度器的状态到指定的路径。

        参数:
            path (str): 保存文件的路径。
        """
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path, restore_optimizer = True):
        """
        从指定的路径加载模型、优化器和学习率调度器的状态。

        参数:
            path (str): 检查点文件的路径。
            restore_optimizer (bool, 可选): 是否恢复优化器和学习率调度器的状态。默认值为 True。
        """
        # 从 Accelerator 中解包模型
        model = self.accelerator.unwrap_model(self.model)
        # 从检查点文件加载模型状态
        pkg = model.load(path)

        if restore_optimizer:
            # 加载优化器的状态
            self.optim.load_state_dict(pkg['optim'])
            # 加载学习率调度器的状态
            self.scheduler.load_state_dict(pkg['scheduler'])

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            # 计算当前的步数，加 1 以避免覆盖最后一个检查点
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        """
        使用 Accelerator 的打印功能打印消息。

        参数:
            msg (str): 要打印的消息。
        """
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        """
        使用模型生成输出。

        参数:
            *args: 传递给模型的生成方法的任意数量的位置参数。
            **kwargs: 传递给模型的生成方法的任意数量的关键字参数。

        返回:
            生成的输出。
        """
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        """
        获取当前设备信息。

        返回:
            torch.device: 当前设备（CPU 或 GPU）。
        """
        return self.accelerator.device

    @property
    def is_distributed(self):
        """
        判断是否在分布式模式下运行。

        返回:
            bool: 如果在分布式模式下运行，则返回 True；否则返回 False。
        """
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        """
        判断当前进程是否为主要的进程。

        返回:
            bool: 如果是主要进程，则返回 True；否则返回 False。
        """
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        """
        判断当前进程是否为本地主要进程。

        返回:
            bool: 如果是本地主要进程，则返回 True；否则返回 False。
        """
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        """
        计算当前步数的学习率，进行学习率预热。

        参数:
            step (int): 当前步数。

        返回:
            float: 当前的学习率。
        """
        if step < self.num_warmup_steps:
            # 如果当前步数小于预热步数，则进行学习率预热
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            # 否则，使用设定的学习率
            return self.lr
    
    def train_step(self):
        """
        执行一个训练步骤，包括前向传播、计算损失、反向传播和优化器更新。
        """
        # 获取当前的训练步数
        steps = int(self.steps.item())

        # 将模型设置为训练模式
        self.model.train()
        
        # adjust the lr according to the schedule
        # 根据学习率调度调整学习率
        if steps < self.num_warmup_steps:
            # Apply warmup
            # 如果当前步数小于预热步数，则进行学习率预热
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                # 更新优化器的学习率
                param_group['lr'] = lr
        else:
            # After warmup period, start to apply CosineAnnealingLR
            # 预热期结束后，开始应用余弦退火学习率调度
            self.scheduler.step()

        # logs
        # 初始化日志字典
        logs = {}

        # update generator
        # 更新生成器

        for _ in range(self.grad_accum_every):
            # 从数据加载器中获取下一个批次的数据
            semantic_token_ids, acoustic_token_ids = next(self.dl_iter)

            # 前向传播，计算损失
            loss, loss_breakdown = self.model(
                acoustic_token_ids,
                cond_semantic_token_ids = semantic_token_ids,
                only_train_generator = self.only_train_generator,
                only_train_critic = self.only_train_critic
            )

            # 解包损失分解
            generator_loss, critic_loss = loss_breakdown
            # 如果生成器损失为 None，则设为 0
            generator_loss = 0. if generator_loss is None else generator_loss
            # 如果批评器损失为 None，则设为 0
            critic_loss = 0. if critic_loss is None else critic_loss
            
            # 反向传播并累积梯度
            self.accelerator.backward(loss / self.grad_accum_every)

            # 累积日志
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every, 'generator_loss': generator_loss / self.grad_accum_every, 'critic_loss': critic_loss / self.grad_accum_every})

        # 梯度裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 更新优化器
        self.optim.step()
        # 清零梯度
        self.optim.zero_grad()

        # log
        # 记录日志
        self.print(f"{steps}: loss: {logs['loss']:0.3f}, generator loss: {logs['generator_loss']:0.3f}, critic loss: {logs['critic_loss']:0.3f}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often
        # 每隔一定的步数进行采样并记录结果
        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            # 从验证数据加载器中获取下一个批次的数据
            semantic_token_ids, acoustic_token_ids = next(self.valid_dl_iter)

            with torch.inference_mode():
                # 将模型设置为评估模式
                self.model.eval() 
                # 前向传播，计算验证损失
                valid_loss, valid_loss_breakdown = self.model(acoustic_token_ids, cond_semantic_token_ids = semantic_token_ids)
                
                # 解包验证损失分解
                valid_generator_loss, valid_critic_loss = valid_loss_breakdown
                # 如果生成器验证损失为 None，则设为 0
                valid_generator_loss = 0. if valid_generator_loss is None else valid_generator_loss
                # 如果批评器验证损失为 None，则设为 0
                valid_critic_loss = 0. if valid_critic_loss is None else valid_critic_loss

            # 打印验证损失
            self.print(f'{steps}: valid loss {valid_loss:0.3f}, valid generator loss {valid_generator_loss:0.3f}, valid critic loss {valid_critic_loss:0.3f}')
            # 记录验证损失到日志
            self.accelerator.log({"valid_loss": valid_loss, "valid_generator_loss": valid_generator_loss, "valid_critic_loss": valid_critic_loss}, step=steps)

        # save model every so often
        # 每隔一定的步数保存模型

        if self.is_main and not (steps % self.save_model_every):
            # 定义模型保存路径
            model_path = str(self.results_folder / f'soundstorm.{steps}.pt')
            # 保存模型
            self.save(model_path)

            # 打印保存信息
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        # 增加步数
        self.steps += 1
        # 返回日志
        return logs

    def train(self, log_fn = noop):
        """
        执行完整的训练过程，直到达到总训练步数。

        参数:
            log_fn (callable, 可选): 日志记录函数。默认值为 noop（无操作）。
        """
        while self.steps < self.num_train_steps:
            # 执行一个训练步骤
            logs = self.train_step()
            # 调用日志记录函数
            log_fn(logs)

        # 打印训练完成信息
        self.print('training complete')
