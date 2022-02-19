# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import wandb
import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from . import logger as log
from . import utils
from .logger import TrainingMetrics, ValidationMetrics
from .models.common import EMA

def binary_accuracy(logit, y, apply_sigmoid=True, reduce=True) -> float:
    prob = logit.sigmoid() if apply_sigmoid else logit
    pred = prob.round().long().view(-1)
    return (pred == y).float().mean() if reduce else (pred == y).float()


def multiclass_accuracy(scores, y, reduce=True):
    _, pred = scores.max(dim=1)
    return (pred == y).float().mean() if reduce else (pred == y).float()


def multiclass_accuracies(scores, y, tops: Tuple[int]):
    _, pred = scores.topk(k=max(tops), dim=1)
    labelled = (y != -100)
    if not any(labelled):
        return [1.0 for i in tops]
    hit = (pred[labelled] == y[labelled, None])
    topk_acc = hit.float().cumsum(dim=1).mean(dim=0)
    return [topk_acc[i-1] for i in tops]

class _Loss:
    def __call__(self, *args):
        raise NotImplementedError

    def metrics(self, *args):
        "Returns list. First element is used for comparison (higher = better)"
        raise NotImplementedError

    def trn_metrics(self):
        "Metrics from last call."
        raise NotImplementedError

    metric_names = []

    
class _MultiExitAccuracy(_Loss):
    def __init__(self, n_exits, acc_tops=(1,), _binary_clf=False):
        self.n_exits = n_exits
        self._binary_clf = _binary_clf
        self._acc_tops = acc_tops
        self._cache = dict()
        self.metric_names = [f'acc{i}_avg' for i in acc_tops]
        for i in acc_tops:
            self.metric_names += [f'acc{i}_clf{k}' for k in range(n_exits)]
            self.metric_names += [f'acc{i}_ens{k}' for k in range(1, n_exits)]
        self.metric_names += ['avg_maxprob']

    def __call__(self, *args):
        raise NotImplementedError

    def _metrics(self, logits_list, y):
        ensemble = torch.zeros_like(logits_list[0])
        acc_clf = np.zeros((self.n_exits, len(self._acc_tops)))
        acc_ens = np.zeros((self.n_exits, len(self._acc_tops)))
        
        for i, logits in enumerate(logits_list):
            if self._binary_clf:
                ensemble = ensemble*i/(i+1) + F.sigmoid(logits)/(i+1)
                acc_clf[i] = binary_accuracy(logits, y)
                acc_ens[i] = binary_accuracy(ensemble, y, apply_sigmoid=False)
            else:
                ensemble += F.softmax(logits, dim=1)
                acc_clf[i] = multiclass_accuracies(logits, y, self._acc_tops)
                acc_ens[i] = multiclass_accuracies(ensemble, y, self._acc_tops)
                
        maxprob = F.softmax(logits_list[-1].data, dim=1).max(dim=1)[0].mean()

        out = list(acc_clf.mean(axis=0))
        for i in range(acc_clf.shape[1]):
            out += list(acc_clf[:, i])
            out += list(acc_ens[1:, i])
        return out + [maxprob]

    def metrics(self, X, y, *args):
        logits_list = X
        return self._metrics(logits_list, y)

    def trn_metrics(self):
        return self._metrics(self._cache['logits_list'], self._cache['y'])
    
    
class ClassificationOnlyLoss(_MultiExitAccuracy):
    def __call__(self, X, y, *args):
        self._cache['logits_list'] = X
        self._cache['y'] = y
        return sum(F.cross_entropy(logits, y)
                   for logits in self._cache['logits_list'])/len(X)


class Executor:
    def __init__(
        self,
        model: nn.Module,
        loss: Optional[nn.Module],
        cuda: bool = True,
        memory_format: torch.memory_format = torch.contiguous_format,
        amp: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        divide_loss: int = 1,
        ts_script: bool = False,
    ):
        assert not (amp and scaler is None), "Gradient Scaler is needed for AMP"

        def xform(m: nn.Module) -> nn.Module:
            if cuda:
                m = m.cuda()
            m.to(memory_format=memory_format)
            return m

        self.model = xform(model)
        if ts_script:
            self.model = torch.jit.script(self.model)
        self.ts_script = ts_script
        # TODO: help!
        self.loss = ClassificationOnlyLoss(len(self.model.exit_list))
        # self.loss = xform(loss) if loss is not None else None
        self.amp = amp
        self.scaler = scaler
        self.is_distributed = False
        self.divide_loss = divide_loss
        self._fwd_bwd = None
        self._forward = None

    def distributed(self, gpu_id):
        self.is_distributed = True
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
        torch.cuda.current_stream().wait_stream(s)

    def _fwd_bwd_fn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with autocast(enabled=self.amp):
            loss = self.loss(self.model(input), target)
            loss /= self.divide_loss

        self.scaler.scale(loss).backward()
        return loss

    def _forward_fn(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast(enabled=self.amp):
            output = self.model(input)
            loss = None if self.loss is None else self.loss(output, target)

        return output if loss is None else loss, output

    def optimize(self, fn):
        return fn

    @property
    def forward_backward(self):
        if self._fwd_bwd is None:
            if self.loss is None:
                raise NotImplementedError(
                    "Loss must not be None for forward+backward step"
                )
            self._fwd_bwd = self.optimize(self._fwd_bwd_fn)
        return self._fwd_bwd

    @property
    def forward(self):
        if self._forward is None:
            self._forward = self.optimize(self._forward_fn)
        return self._forward

    def train(self):
        self.model.train()
        # if self.loss is not None:
        #     self.loss.train()

    def eval(self):
        self.model.eval()
        # if self.loss is not None:
        #     self.loss.eval()


class Trainer:
    def __init__(
        self,
        executor: Executor,
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int,
        ema: Optional[float] = None,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.use_ema = False
        if ema is not None:
            self.ema_executor = deepcopy(self.executor)
            self.ema = EMA(ema, self.ema_executor.model)
            self.use_ema = True

        self.optimizer.zero_grad(set_to_none=True)
        self.steps_since_update = 0

    def train(self):
        self.executor.train()
        if self.use_ema:
            self.ema_executor.train()

    def eval(self):
        self.executor.eval()
        if self.use_ema:
            self.ema_executor.eval()

    def train_step(self, input, target, step=None):
        loss = self.executor.forward_backward(input, target)

        self.steps_since_update += 1

        if self.steps_since_update == self.grad_acc_steps:
            if self.executor.scaler is not None:
                self.executor.scaler.step(self.optimizer)
                self.executor.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0

        torch.cuda.synchronize()

        if self.use_ema:
            self.ema(self.executor.model, step=step)

        return loss

    def validation_steps(self) -> Dict[str, Callable]:
        vsd: Dict[str, Callable] = {"val": self.executor.forward}
        if self.use_ema:
            vsd["val_ema"] = self.ema_executor.forward
        return vsd

    def state_dict(self) -> dict:
        res = {
            "state_dict": self.executor.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_ema:
            res["state_dict_ema"] = self.ema_executor.model.state_dict()

        return res


def train(
    train_step,
    train_loader,
    lr_scheduler,
    log_fn,
    timeout_handler,
    prof=-1,
    step=0,
):
    interrupted = False

    end = time.time()

    data_iter = enumerate(train_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr = lr_scheduler(i)
        data_time = time.time() - end

        loss = train_step(input, target, step=step + i)
        it_time = time.time() - end

        with torch.no_grad():
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.detach())
            else:
                reduced_loss = loss.detach()

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            lr=lr,
            loss=reduced_loss.item(),
        )

        end = time.time()
        if prof > 0 and (i + 1 >= prof):
            time.sleep(5)
            break
        if ((i + 1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break

    return interrupted


def validate(infer_fn, val_loader, exit_list, exit_idx, log_fn, prof=-1, with_loss=True):
    # switch to evaluate mode
    end = time.time()
    last_exit_res = [0, 0]

    data_iter = enumerate(val_loader)
    top1 = log.AverageMeter()
    top5 = log.AverageMeter()
    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        if with_loss:
            loss, output = infer_fn(input, target)
        else:
            output = infer_fn(input)

        with torch.no_grad():
            prec1, prec5 = utils.accuracy(output[exit_idx], target, topk=(1, 5))

            if torch.distributed.is_initialized():
                if with_loss:
                    reduced_loss = utils.reduce_tensor(loss.detach())
                prec1 = utils.reduce_tensor(prec1)
                prec5 = utils.reduce_tensor(prec5)
            else:
                if with_loss:
                    reduced_loss = loss.detach()

        prec1 = prec1.item()
        prec5 = prec5.item()
        infer_result = {
            "top1": (prec1, bs),
            "top5": (prec5, bs),
        }

        if with_loss:
            infer_result["loss"] = (reduced_loss.item(), bs)

        torch.cuda.synchronize()

        it_time = time.time() - end

        top1.record(prec1, bs)
        top5.record(prec5, bs)

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            **infer_result,
        )

        end = time.time()
        if (prof > 0) and (i + 1 >= prof):
            time.sleep(5)
            break
        
        # print(top1.get_val()[0])
        # print(top1.get_val()[1])
    wandb.log({'pred1_e{}'.format(exit_list[exit_idx]):top1.get_val()[0]})
    wandb.log({'pred5_e{}'.format(exit_list[exit_idx]):top5.get_val()[0]})
        # last_exit_res[0] = top1.get_val()[0]
        # last_exit_res[1] = top1.get_val()[1]

    return top1.get_val()


# Train loop {{{
def train_loop(
    trainer: Trainer,
    lr_scheduler,
    train_loader,
    train_loader_len,
    val_loader,
    exit_list,
    logger,
    should_backup_checkpoint,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
):
    train_metrics = TrainingMetrics(logger)
    val_metrics = {
        k: ValidationMetrics(logger, k) for k in trainer.validation_steps().keys()
    }
    training_step = trainer.train_step

    prec1 = -1

    if early_stopping_patience > 0:
        epochs_since_improvement = 0
    backup_prefix = (
        checkpoint_filename[: -len("checkpoint.pth.tar")]
        if checkpoint_filename.endswith("checkpoint.pth.tar")
        else ""
    )

    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    with utils.TimeoutHandler() as timeout_handler:
        interrupted = False
        metric_record = []
        for epoch in range(start_epoch, end_epoch):
            if logger is not None:
                logger.start_epoch()
            if not skip_training:
                if logger is not None:
                    data_iter = logger.iteration_generator_wrapper(
                        train_loader, mode="train"
                    )
                else:
                    data_iter = train_loader

                trainer.train()
                interrupted = train(
                    training_step,
                    data_iter,
                    lambda i: lr_scheduler(trainer.optimizer, i, epoch),
                    train_metrics.log,
                    timeout_handler,
                    prof=prof,
                    step=epoch * train_loader_len,
                )

            if not skip_validation:
                trainer.eval()
                prec1 = 0
                for exit_idx in range(len(exit_list)):
                    for k, infer_fn in trainer.validation_steps().items():
                        if logger is not None:
                            data_iter = logger.iteration_generator_wrapper(
                                val_loader, mode="val"
                            )
                        else:
                            data_iter = val_loader

                    
                        step_prec1, _ = validate(
                            infer_fn,
                            data_iter,
                            exit_list,
                            exit_idx,
                            val_metrics[k].log,
                            prof=prof,
                        )

                        if k == "val":
                            prec1 = step_prec1

                metric_record.append(prec1)
                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = False
                best_prec1 = 0

            if logger is not None:
                logger.end_epoch()

            if save_checkpoints and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                if should_backup_checkpoint(epoch):
                    backup_filename = "{}checkpoint-{}.pth.tar".format(
                        backup_prefix, epoch + 1
                    )
                else:
                    backup_filename = None
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "best_prec1": best_prec1,
                    **trainer.state_dict(),
                }
                utils.save_checkpoint(
                    checkpoint_state,
                    is_best,
                    checkpoint_dir=checkpoint_dir,
                    backup_filename=backup_filename,
                    filename=checkpoint_filename,
                )

            # diff = [metric_record[i+1]-metric_record[i] for i in range(len(metric_record)-1)]
            # if len([i for i in diff[-3:] if i < 0.2]) == 3:
            #     break

            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            if interrupted:
                break


# }}}
