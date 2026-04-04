# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""


from __future__ import annotations

import sys
import os
import argparse





import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from torch.distributed.rpc.api import method_name

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion#这一块进cuda初始化，进入watersplatting模型
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.configs.method_configs import all_methods,all_descriptions

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """
        主训练函数，每个进程调用该函数进行训练设置并运行训练器。

        Args:
            local_rank (int): 当前进程在本地机器中的 GPU ID（进程内编号）。
            world_size (int): 总的 GPU 数量（跨机器的全局 GPU 总数）。
            config (TrainerConfig): 用于指定训练流程的配置文件。
            global_rank (int): 当前进程在全局范围内的 ID，默认为 0。
        """
    _set_random_seed(config.machine.seed + global_rank)# 设置随机种子，确保每个进程的随机性独立可控。
    trainer = config.setup(local_rank=local_rank, world_size=world_size)# 使用配置对象初始化 Trainer 实例。
    trainer.setup()# 设置 Trainer
    trainer.train()# 开始训练。


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """
    分布式训练的工作进程函数。每个 GPU/进程会调用一次，用于初始化进程组并执行训练。

    Args:
        local_rank (int): 当前进程在本地机器中的 GPU ID（进程内编号）。
        main_func (Callable): 每个分布式工作进程将调用的主函数（通常是 `train_loop`）。
        world_size (int): 总的 GPU 数量（跨机器的全局 GPU 总数）。
        num_devices_per_machine (int): 每台机器的 GPU 数量。
        machine_rank (int): 当前机器的编号（从 0 开始）。
        dist_url (str): 分布式训练的初始化方法及 URL，例如 `tcp://127.0.0.1:8686`。
            如果设置为 "auto"，会自动选择一个本地空闲端口。
        config (TrainerConfig): 用于指定训练流程的配置文件。
        timeout (timedelta): 分布式进程组的初始化超时时间，默认 30 分钟。
        device_type (Literal["cpu", "cuda", "mps"]): 使用的设备类型，默认 "cuda"。

    Raises:
        AssertionError: 如果 CUDA 不可用。
        Exception: 如果分布式进程组初始化失败。

    Returns:
        Any: 主函数（`main_func`）的返回值。
    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."# 检查 CUDA 是否可用。
    global_rank = machine_rank * num_devices_per_machine + local_rank# 计算全局进程编号（基于机器编号和本地 GPU ID）。

    dist.init_process_group(# 初始化分布式进程组。
        backend="nccl" if device_type == "cuda" else "gloo",  # 使用 NCCL（GPU）或 Gloo（CPU）作为通信后端。
        init_method=dist_url,  # 分布式进程组的初始化 URL。
        world_size=world_size,  # 总进程数。
        rank=global_rank,  # 当前进程的全局编号。
        timeout=timeout,  # 超时时间。
    )
    assert comms.LOCAL_PROCESS_GROUP is None# 确保本地进程组尚未初始化。
    num_machines = world_size // num_devices_per_machine# 计算总机器数量。
    for i in range(num_machines): # 为每台机器的进程创建本地通信组（用于跨 GPU 通信）
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))# 获取第 i 台机器的所有进程编号。
        pg = dist.new_group(ranks_on_i)# 为该机器创建一个新的通信组。
        if i == machine_rank:# 如果是当前机器，则设置本地通信组。
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()# 确保本地 GPU 数量足够。
    output = main_func(local_rank, world_size, config, global_rank)# 调用主函数
    comms.synchronize()# 同步所有进程（确保所有操作完成）。
    dist.destroy_process_group()# 销毁进程组，释放资源。
    return output# 返回主函数的输出结果。


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """
    此函数用于生成多个进程以调用主函数 main_func
    参数：
        main_func (Callable): 由分布式工作进程调用的函数
        num_devices_per_machine (int): 每台机器上的 GPU 数量
        num_machines (int, 可选): 总机器数量
        machine_rank (int, 可选): 当前机器的排名
        dist_url (str, 可选): 用于分布式任务连接的 URL
        config (TrainerConfig, 可选): 指定训练方案的配置文件
        timeout (timedelta, 可选): 分布式工作进程的超时时间
        device_type: 用于训练的设备类型
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig) -> None:
    """Main function."""

    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Run the script with dataset-specific configurations.")

    # Add --config argument to accept the YAML configuration file
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')

    # Parse the arguments
    return parser.parse_args()



if __name__ == "__main__":

    # Parse command line arguments
    args = parse_args()

    # Load the configuration from the YAML file
    config = load_config(args.config)

    # Start building sys.argv
    sys.argv = [
        "train.py",
        config['model_name'],  # e.g. "lowlight_underwater"
        "--output-dir", config['output_dir'],
        "--vis", config['vis'],
    ]

    # Checking and adding model-specific settings
    if 'pipeline' in config and 'model' in config['pipeline']:
        model_settings = config['pipeline']['model']
        for key, value in model_settings.items():
            # Construct the argument name based on the model setting key
            argument_name = f"--pipeline.model.{key.replace('_', '-')}"
            # Append the argument name and its corresponding value to sys.argv
            sys.argv.extend([argument_name, str(value)])

    # Add the visualization system and its parameters
    sys.argv.append(config['visualization_system'])

    # Add visualization system-specific parameters
    sys.argv.extend([
        "--downscale-factor", str(config['downscale_factor']),
        "--colmap-path", config['colmap_path'],
        "--images-path", config['images_path'],
    ])

    # Print to check the final sys.argv for debugging
    print("Final sys.argv:", sys.argv)

    entrypoint()
