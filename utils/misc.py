import collections
import datetime
import os
import random
import subprocess
import time
from multiprocessing import JoinableQueue, Process

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info

from utils.logger import get_root_logger

os.environ["MOX_SILENT_MODE"] = "1"  # mute moxing log


def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SimpleTimer:
    def __init__(self, num_tasks, log_interval=1, desc="Process"):
        self.num_tasks = num_tasks
        self.desc = desc
        self.count = 0
        self.log_interval = log_interval
        self.start_time = time.time()
        self.logger = get_root_logger()

    def log(self):
        self.count += 1
        if (self.count % self.log_interval) == 0 or self.count == self.num_tasks:
            time_elapsed = time.time() - self.start_time
            avg_time = time_elapsed / self.count
            eta_sec = avg_time * (self.num_tasks - self.count)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            elapsed_str = str(datetime.timedelta(seconds=int(time_elapsed)))
            log_info = f"{self.desc} [{self.count}/{self.num_tasks}], elapsed_time:{elapsed_str}," \
                       f" avg_time: {avg_time}, eta: {eta_str}."
            self.logger.info(log_info)


class DebugUnderflowOverflow:
    """
    This debug class helps detect and understand where the model starts getting very large or very small, and more
    importantly `nan` or `inf` weight and activation elements.
    There are 2 working modes:
    1. Underflow/overflow detection (default)
    2. Specific batch absolute min/max tracing without detection
    Mode 1: Underflow/overflow detection
    To activate the underflow/overflow detection, initialize the object with the model :
    ```python
    debug_overflow = DebugUnderflowOverflow(model)
    ```
    then run the training as normal and if `nan` or `inf` gets detected in at least one of the weight, input or
    output elements this module will throw an exception and will print `max_frames_to_save` frames that lead to this
    event, each frame reporting
    1. the fully qualified module name plus the class name whose `forward` was run
    2. the absolute min and max value of all elements for each module weights, and the inputs and output
    For example, here is the header and the last few frames in detection report for `google/mt5-small` run in fp16 mixed precision :
    ```
    Detected inf/nan during batch_number=0
    Last 21 forward frames:
    abs min  abs max  metadata
    [...]
                      encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
    2.17e-07 4.50e+00 weight
    1.79e-06 4.65e+00 input[0]
    2.68e-06 3.70e+01 output
                      encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
    8.08e-07 2.66e+01 weight
    1.79e-06 4.65e+00 input[0]
    1.27e-04 2.37e+02 output
                      encoder.block.2.layer.1.DenseReluDense.wo Linear
    1.01e-06 6.44e+00 weight
    0.00e+00 9.74e+03 input[0]
    3.18e-04 6.27e+04 output
                      encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
    1.79e-06 4.65e+00 input[0]
    3.18e-04 6.27e+04 output
                      encoder.block.2.layer.1.dropout Dropout
    3.18e-04 6.27e+04 input[0]
    0.00e+00      inf output
    ```
    You can see here, that `T5DenseGatedGeluDense.forward` resulted in output activations, whose absolute max value
    was around 62.7K, which is very close to fp16's top limit of 64K. In the next frame we have `Dropout` which
    renormalizes the weights, after it zeroed some of the elements, which pushes the absolute max value to more than
    64K, and we get an overlow.
    As you can see it's the previous frames that we need to look into when the numbers start going into very large for
    fp16 numbers.
    The tracking is done in a forward hook, which gets invoked immediately after `forward` has completed.
    By default the last 21 frames are printed. You can change the default to adjust for your needs. For example :
    ```python
    debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
    ```
        To validate that you have set up this debugging feature correctly, and you intend to use it in a training that may
        take hours to complete, first run it with normal tracing enabled for one of a few batches as explained in the next
        section.
        Mode 2. Specific batch absolute min/max tracing without detection
        The second work mode is per-batch tracing with the underflow/overflow detection feature turned off.
        Let's say you want to watch the absolute min and max values for all the ingredients of each `forward` call of a
    given batch, and only do that for batches 1 and 3. Then you instantiate this class as :
    ```python
    debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1,3])
    ```
    And now full batches 1 and 3 will be traced using the same format as explained above. Batches are 0-indexed.
    This is helpful if you know that the program starts misbehaving after a certain batch number, so you can
    fast-forward right to that area.
    Early stopping:
    You can also specify the batch number after which to stop the training, with :
    ```python
    debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1,3], abort_after_batch_num=3)
    ```
    This feature is mainly useful in the tracing mode, but you can use it for any mode.
    **Performance**:
    As this module measures absolute `min`/``max` of each weight of the model on every forward it'll slow the
    training down. Therefore remember to turn it off once the debugging needs have been met.
    Args:
        model (`nn.Module`):
            The model to debug.
        max_frames_to_save (`int`, *optional*, defaults to 21):
            How many frames back to record
        trace_batch_nums(`List[int]`, *optional*, defaults to `[]`):
            Which batch numbers to trace (turns detection off)
        abort_after_batch_num  (`int``, *optional*):
            Whether to abort after a certain batch number has finished
    """

    def __init__(self, model, max_frames_to_save=21, trace_batch_nums=None, abort_after_batch_num=None):
        if trace_batch_nums is None:
            trace_batch_nums = []
        self.model = model
        self.trace_batch_nums = trace_batch_nums
        self.abort_after_batch_num = abort_after_batch_num

        # keep a LIFO buffer of frames to dump as soon as inf/nan is encountered to give context to the problem emergence
        self.frames = collections.deque([], max_frames_to_save)
        self.frame = []
        self.batch_number = 0
        self.total_calls = 0
        self.detected_overflow = False
        self.prefix = "                 "

        self.analyse_model()

        self.register_forward_hook()

    def save_frame(self, frame=None):
        if frame is not None:
            self.expand_frame(frame)
        self.frames.append("\n".join(self.frame))
        self.frame = []  # start a new frame

    def expand_frame(self, line):
        self.frame.append(line)

    def trace_frames(self):
        print("\n".join(self.frames))
        self.frames = []

    def reset_saved_frames(self):
        self.frames = []

    def dump_saved_frames(self):
        print(f"\nDetected inf/nan during batch_number={self.batch_number} "
              f"Last {len(self.frames)} forward frames:"
              f"{'abs min':8} {'abs max':8} metadata"
              f"'\n'.join(self.frames)"
              f"\n\n")
        self.frames = []

    def analyse_model(self):
        # extract the fully qualified module names, to be able to report at run time. e.g.:
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # for shared weights only the first shared module name will be registered
        self.module_names = {m: name for name, m in self.model.named_modules()}
        # self.longest_module_name = max(len(v) for v in self.module_names.values())

    def analyse_variable(self, var, ctx):
        if torch.is_tensor(var):
            self.expand_frame(self.get_abs_min_max(var, ctx))
            if self.detect_overflow(var, ctx):
                self.detected_overflow = True
        elif var is None:
            self.expand_frame(f"{'None':>17} {ctx}")
        else:
            self.expand_frame(f"{'not a tensor':>17} {ctx}")

    def batch_start_frame(self):
        self.expand_frame(f"\n\n{self.prefix} *** Starting batch number={self.batch_number} ***")
        self.expand_frame(f"{'abs min':8} {'abs max':8} metadata")

    def batch_end_frame(self):
        self.expand_frame(f"{self.prefix} *** Finished batch number={self.batch_number - 1} ***\n\n")

    def create_frame(self, module, input, output):
        self.expand_frame(f"{self.prefix} {self.module_names[module]} {module.__class__.__name__}")

        # params
        for name, p in module.named_parameters(recurse=False):
            self.analyse_variable(p, name)

        # inputs
        if isinstance(input, tuple):
            for i, x in enumerate(input):
                self.analyse_variable(x, f"input[{i}]")
        else:
            self.analyse_variable(input, "input")

        # outputs
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # possibly a tuple of tuples
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        self.analyse_variable(y, f"output[{i}][{j}]")
                else:
                    self.analyse_variable(x, f"output[{i}]")
        else:
            self.analyse_variable(output, "output")

        self.save_frame()

    def register_forward_hook(self):
        self.model.apply(self._register_forward_hook)

    def _register_forward_hook(self, module):
        module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        # - input is a tuple of packed inputs (could be non-Tensors)
        # - output could be a Tensor or a tuple of Tensors and non-Tensors

        last_frame_of_batch = False

        trace_mode = self.batch_number in self.trace_batch_nums
        if trace_mode:
            self.reset_saved_frames()

        if self.total_calls == 0:
            self.batch_start_frame()
        self.total_calls += 1

        # count batch numbers - the very first forward hook of the batch will be called when the
        # batch completes - i.e. it gets called very last - we know this batch has finished
        if module == self.model:
            self.batch_number += 1
            last_frame_of_batch = True

        self.create_frame(module, input, output)

        # if last_frame_of_batch:
        #     self.batch_end_frame()

        if trace_mode:
            self.trace_frames()

        if last_frame_of_batch:
            self.batch_start_frame()

        if self.detected_overflow and not trace_mode:
            self.dump_saved_frames()

            # now we can abort, as it's pointless to continue running
            raise ValueError(
                "DebugUnderflowOverflow: inf/nan detected, aborting as there is no point running further. "
                "Please scroll up above this traceback to see the activation values prior to this event."
            )

        # abort after certain batch if requested to do so
        if self.abort_after_batch_num is not None and self.batch_number > self.abort_after_batch_num:
            raise ValueError(
                f"DebugUnderflowOverflow: aborting after {self.batch_number} batches due to `abort_after_batch_num={self.abort_after_batch_num}` arg"
            )

    @staticmethod
    def get_abs_min_max(var, ctx):
        abs_var = var.abs()
        return f"{abs_var.min():8.2e} {abs_var.max():8.2e} {ctx}"

    @staticmethod
    def detect_overflow(var, ctx):
        """
        Report whether the tensor contains any `nan` or `inf` entries.
        This is useful for detecting overflows/underflows and best to call right after the function that did some math that
        modified the tensor in question.
        This function contains a few other helper features that you can enable and tweak directly if you want to track
        various other things.
        Args:
            var: the tensor variable to check
            ctx: the message to print as a context
        Return:
            `True` if `inf` or `nan` was detected, `False` otherwise
        """
        detected = False
        if torch.isnan(var).any().item():
            detected = True
            print(f"{ctx} has nans")
        if torch.isinf(var).any().item():
            detected = True
            print(f"{ctx} has infs")
        if var.dtype == torch.float32 and torch.ge(var.abs(), 65535).any().item():
            detected = True
            print(f"{ctx} has overflow values {var.abs().max().item()}.")
        return detected
