# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The KERMT trainer.
"""
import os
import time
from logging import Logger
from typing import List, Tuple
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from kermt.model.layers import dynamic_depth_can_skip_message_passing
try:
    import wandb
except ImportError:
    wandb = None


# ============================================================================
# Shared helper functions for checkpoint saving/loading
# ============================================================================

def save_checkpoint(model, optimizer, args, batch_idx, n_steps, epoch, 
                   file_path, name=None, save_last=False) -> str:
    """
    Save model checkpoint. Shared by both KERMTTrainer and KERMTCMIMTrainer.
    
    :param model: the model (wrapped in DDP)
    :param optimizer: the optimizer
    :param args: training arguments
    :param batch_idx: current batch index
    :param n_steps: current step number
    :param epoch: current epoch
    :param file_path: directory to save checkpoint
    :param name: optional custom filename
    :param save_last: whether to also save as 'last_checkpoint.pt'
    :return: path to saved checkpoint
    """
    now = time.localtime()
    if name is None:
        name = "_%04d_%02d_%02d_%02d_%02d_%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    output_path = os.path.join(file_path, name)
    
    scaler = None
    features_scaler = None
    # Capture WandB run ID so restarts can resume the same run
    wandb_run_id = None
    if wandb is not None and wandb.run is not None:
        wandb_run_id = wandb.run.id

    state = {
        'args': args,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler_step': n_steps,
        'batch_idx': batch_idx,
        "epoch": epoch,
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None,
        'wandb_run_id': wandb_run_id,
    }
    # Use atomic save pattern: write to .tmp then rename
    # This prevents corrupted checkpoints if the process is killed mid-write
    tmp_output_path = output_path + ".tmp"
    torch.save(state, tmp_output_path)
    os.replace(tmp_output_path, output_path)  # atomic on POSIX
    
    if save_last:
        last_path = os.path.join(file_path, "last_checkpoint.pt")
        tmp_last_path = last_path + ".tmp"
        torch.save(state, tmp_last_path)
        os.replace(tmp_last_path, last_path)  # atomic on POSIX

    print(f"Model at step={n_steps} saved at {output_path}", flush=True)
    return output_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler) -> Tuple[int, int, int, ...]:
    """
    Load model checkpoint. Shared by both KERMTTrainer and KERMTCMIMTrainer.

    :param checkpoint_path: path to checkpoint file
    :param model: the model (wrapped in DDP)
    :param optimizer: the optimizer
    :param scheduler: the learning rate scheduler
    :return: tuple of (epoch, scheduler_step, batch_idx, wandb_run_id)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found")
        return 0, 0, 0, None

    # TODO(sveccham): Change this to weights_only=True
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.module.load_state_dict(ckpt["state_dict"])

    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"Could not load optimizer state ({e}); starting with fresh optimizer state.", flush=True)
    else:
        print("Checkpoint has no optimizer state (slim release ckpt); starting with fresh optimizer state.", flush=True)

    scheduler_step = ckpt.get("scheduler_step", 0)
    scheduler.current_step = scheduler_step
    epoch = ckpt.get("epoch", 0)
    batch_idx = ckpt.get("batch_idx", 0)
    wandb_run_id = ckpt.get("wandb_run_id", None)

    print(f"Batch index from loaded checkpoint: {batch_idx}", flush=True)
    return epoch, scheduler_step, batch_idx, wandb_run_id


def _wandb_log(args, gpu_id, log_dict, step=None):
    """Log metrics to WandB if enabled and on rank 0."""
    if gpu_id == 0 and wandb is not None and getattr(args, 'wandb_project', None):
        wandb.log(log_dict, step=step)


class KERMTTrainer:
    def __init__(self,
                 args,
                 model: Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer,
                 scheduler,
                 gpu_id,
                 n_steps: int,
                 logger: Logger = None,
                 shutdown_checker=None):
        """
        The init function of KERMTTrainer
        :param args: the input arguments
        :param model: the complete KermtTask model (created externally)
        :param train_dataloader: the training dataloader
        :param val_dataloader: the validation dataloader
        :param optimizer: the optimizer (built on model.parameters())
        :param scheduler: the scheduler
        :param gpu_id: the gpu id
        :param n_steps: initial step count
        :param logger: the logger
        :param shutdown_checker: callable that returns True if graceful shutdown requested
        """

        self.args = args
        self.model = model
        self.loss_func = self.model.get_loss_func(args)
        self.gpu_id = gpu_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.debug = logger.debug if logger is not None else print
        self.shutdown_checker = shutdown_checker  # For graceful shutdown on cluster time limits

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_iter = 0

        self.model.to(self.gpu_id)

        # find_unused_parameters is required in two cases:
        #  * embedding_output_type != 'both': in atom/bond mode the encoder produces only
        #    one aggregation level, so the opposite branch's FFNs and the bond/atom vocab +
        #    FG heads receive no gradient.
        #  * a small --depth: the dyMPN depth draw can then land low enough to skip message
        #    passing entirely, leaving W_h without gradient on that batch. See
        #    dynamic_depth_can_skip_message_passing().
        # Neither holds for the usual 'both' + --depth 6, so the flag stays off there and
        # DDP keeps its fast path.
        # (static_graph is not an option: dynamic-depth sampling varies the graph per step.)
        self.model = DDP(self.model, device_ids=[gpu_id],
                         find_unused_parameters=(self.args.embedding_output_type != 'both'
                                                 or dynamic_depth_can_skip_message_passing(self.args.depth)))

        if self.args.tensorboard:
            self.writer = SummaryWriter(self.args.save_dir)

        # Var by SV (not sure what n_iter is doing)
        self.n_steps = n_steps
        self.first_epoch_post_resume = True
        self.curr_epoch_batch_idx = 0 # Number of batches to skip in first epoch of current training
        self.batch_idx_offset = 0  # Offset to add when saving batch_idx (for sampler-level resume)

    def train(self, start_epoch: int, max_epochs: int) -> List:
        """
        The training iteration
        :param max_epochs: the max epochs.
        :return: the loss terms of current epoch.
        """
        for epoch in range(start_epoch, max_epochs):
            s_time = time.time()
            _, train_loss, _ = self.iter(epoch, train=True)
            t_time = time.time() - s_time
            if self.gpu_id == 0:
                print(f"epoch={epoch:04d}, cur_lr={self.scheduler.get_lr()[0]:.5f}, train_loss={train_loss:.6f}, train_time={t_time:.2f}", flush=True)
            # After the resumed epoch completes, reset sampler and offset so
            # subsequent epochs iterate over all samples from the beginning
            if epoch == start_epoch and hasattr(self.train_dataloader.sampler, 'set_start_index'):
                self.train_dataloader.sampler.set_start_index(0)
                self.batch_idx_offset = 0

    def validation(self, max_val_batches: int) -> List:
        """
        The validation iteration
        :param max_val_batches: the maximum number of batches to validate.
        :return: the loss terms as a list
        """
        self.model.eval()
        loss_sum, iter_count = 0, 0
        n_batches = 0
        av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum, bv_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0, 0, 0
        # loss_func = self.model.get_loss_func(self.args)

        for ibatch, item in enumerate(self.val_dataloader):
            batch_graph = item["graph_input"]
            targets = item["targets"]
            targets["av_task"] = targets["av_task"].to(self.gpu_id)
            targets["bv_task"] = targets["bv_task"].to(self.gpu_id)
            targets["fg_task"] = targets["fg_task"].to(self.gpu_id)

            preds = self.model(batch_graph)

            loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = self.loss_func(preds, targets)

            loss_sum += loss.item()
            iter_count += self.args.batch_size

            av_loss_sum += av_loss.item() if not isinstance(av_loss, float) else av_loss
            bv_loss_sum += bv_loss.item() if not isinstance(bv_loss, float) else bv_loss
            fg_loss_sum += fg_loss.item() if not isinstance(fg_loss, float) else fg_loss
            av_dist_loss_sum += av_dist_loss.item() if type(av_dist_loss) != float else av_dist_loss
            bv_dist_loss_sum += bv_dist_loss.item() if type(bv_dist_loss) != float else bv_dist_loss
            fg_dist_loss_sum += fg_dist_loss.item() if type(fg_dist_loss) != float else fg_dist_loss

            n_batches += 1
            if n_batches >= max_val_batches:
                break

        # Compute per batch losses
        loss_sum /= n_batches
        av_loss_sum /= n_batches
        bv_loss_sum /= n_batches
        fg_loss_sum /= n_batches
        av_dist_loss_sum /= n_batches
        bv_dist_loss_sum /= n_batches
        fg_dist_loss_sum /= n_batches


        if self.gpu_id == 0:
            print(f"Validation loss: {loss_sum:.4f}, av_loss: {av_loss_sum:.4f}, bv_loss: {bv_loss_sum:.4f}, fg_loss: {fg_loss_sum:.4f}", flush=True)
            val_metrics = {
                'val/loss': loss_sum,
                'val/av_loss': av_loss_sum,
                'val/bv_loss': bv_loss_sum,
                'val/fg_loss': fg_loss_sum,
            }
            if self.args.tensorboard:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(k, v, self.n_steps)
            _wandb_log(self.args, self.gpu_id, val_metrics, step=self.n_steps)

        self.model.train()
        return loss_sum

    def test(self, epoch: int) -> List:
        """
        The test/validaiion iteration
        :param epoch: the current epoch number.
        :return:  the loss terms as a list
        """
        # return self.mock_iter(epoch, self.test_data, train=False)
        return self.iter(epoch, self.test_data, train=False)

    def mock_iter(self, epoch: int, data_loader: DataLoader, train: bool = True) -> List:
        """
        Perform a mock iteration. For test only.
        :param epoch: the current epoch number.
        :param data_loader: the data loader.
        :param train: True: train model, False: validation model.
        :return: the loss terms as a list
        """

        for _, _ in enumerate(data_loader):
            self.scheduler.step()
        cum_loss_sum = 0.0
        self.n_iter += self.args.batch_size
        return self.n_iter, cum_loss_sum, (0, 0, 0, 0, 0, 0)

    def set_batch_idx(self, batch_idx: int, batch_idx_offset: int = 0):
        """
        Set batch index for resume logic.
        
        Args:
            batch_idx: Number of batches to skip in training loop (0 if sampler handles skipping)
            batch_idx_offset: Offset to add when saving batch_idx (for sampler-level resume)
        """
        self.curr_epoch_batch_idx = batch_idx
        self.batch_idx_offset = batch_idx_offset

    def iter(self, epoch, train=True) -> List:
        """
        Perform a training / validation iteration.
        :param epoch: the current epoch number.
        :param data_loader: the data loader.
        :param train: True: train model, False: validation model.
        :return: the loss terms as a list
        """
        if train:
            self.model.train()
            self.train_dataloader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        loss_sum, iter_count = 0, 0
        cum_loss_sum, cum_iter_count = 0, 0
        av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum, bv_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0, 0, 0
        # loss_func = self.model.get_loss_func(self.args)

        for ibatch, item in enumerate(self.train_dataloader):
            if self.first_epoch_post_resume:
                if ibatch < self.curr_epoch_batch_idx:
                    print(f"Skipping batch {ibatch} because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", flush=True)
                    continue
                elif ibatch >= self.curr_epoch_batch_idx:
                    print(f"Stop skipping batches because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", flush=True)
                    self.first_epoch_post_resume = False

            if self.gpu_id == 0:
                print(f"{self.n_steps=}", flush=True)
            batch_graph = item["graph_input"]
            targets = item["targets"]
            # if next(self.model.parameters()).is_cuda:
            targets["av_task"] = targets["av_task"].to(self.gpu_id)
            targets["bv_task"] = targets["bv_task"].to(self.gpu_id)
            targets["fg_task"] = targets["fg_task"].to(self.gpu_id)

            preds = self.model(batch_graph)

            loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = self.loss_func(preds, targets)

            loss_sum += loss.item()
            iter_count += self.args.batch_size

            if train:
                cum_loss_sum += loss.item()
                # Run model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                # For eval model, only consider the loss of three task.
                cum_loss_sum += av_loss.item() if not isinstance(av_loss, float) else av_loss
                cum_loss_sum += bv_loss.item() if not isinstance(bv_loss, float) else bv_loss
                cum_loss_sum += fg_loss.item() if not isinstance(fg_loss, float) else fg_loss

            av_loss_sum += av_loss.item() if not isinstance(av_loss, float) else av_loss
            bv_loss_sum += bv_loss.item() if not isinstance(bv_loss, float) else bv_loss
            fg_loss_sum += fg_loss.item() if not isinstance(fg_loss, float) else fg_loss
            av_dist_loss_sum += av_dist_loss.item() if type(av_dist_loss) != float else av_dist_loss
            bv_dist_loss_sum += bv_dist_loss.item() if type(bv_dist_loss) != float else bv_dist_loss
            fg_dist_loss_sum += fg_dist_loss.item() if type(fg_dist_loss) != float else fg_dist_loss

            # Save model (batch_idx includes offset for sampler-level resume)
            if (self.gpu_id == 0)and (self.n_steps % self.args.save_interval) == 0:
                self.save(batch_idx=ibatch + self.batch_idx_offset, n_steps=self.n_steps, epoch=epoch, file_path=self.args.save_dir, name=f"model_step_{self.n_steps}.pt", save_last=True)
                if self.args.tensorboard:
                    self.writer.flush()

            # Check for graceful shutdown (e.g., cluster time limit approaching)
            if self.shutdown_checker is not None and self.shutdown_checker():
                if self.gpu_id == 0:
                    # Save checkpoint before exiting
                    self.save(batch_idx=ibatch + self.batch_idx_offset, n_steps=self.n_steps, epoch=epoch, 
                              file_path=self.args.save_dir, name=f"model_step_{self.n_steps}_shutdown.pt", save_last=True)
                    if self.args.tensorboard:
                        self.writer.flush()
                        self.writer.close()
                    print(f"[SHUTDOWN] Graceful shutdown complete. Saved at step {self.n_steps}, batch {ibatch + self.batch_idx_offset}", flush=True)
                raise SystemExit(0)

            cum_iter_count += 1
            self.n_iter += self.args.batch_size
            self.n_steps += 1

            train_log_interval = max(1, self.args.train_interval)
            if self.gpu_id == 0 and self.n_steps % train_log_interval == 0:
                train_metrics = {
                    'train/loss': loss.item(),
                    'train/av_loss': av_loss.item() if not isinstance(av_loss, float) else av_loss,
                    'train/bv_loss': bv_loss.item() if not isinstance(bv_loss, float) else bv_loss,
                    'train/fg_loss': fg_loss.item() if not isinstance(fg_loss, float) else fg_loss,
                    'train/av_dist_loss': av_dist_loss.item() if not isinstance(av_dist_loss, float) else av_dist_loss,
                    'train/bv_dist_loss': bv_dist_loss.item() if not isinstance(bv_dist_loss, float) else bv_dist_loss,
                    'train/fg_dist_loss': fg_dist_loss.item() if not isinstance(fg_dist_loss, float) else fg_dist_loss,
                    'train/lr': self.scheduler.get_lr()[0],
                    'train/epoch': epoch,
                    'train/batch_idx': ibatch + self.batch_idx_offset,
                }
                if self.args.tensorboard:
                    for k, v in train_metrics.items():
                        self.writer.add_scalar(k, v, self.n_steps)
                _wandb_log(self.args, self.gpu_id, train_metrics, step=self.n_steps)
            # Optional: run validation every val_interval steps (similar to train metrics / save_interval)
            val_interval = getattr(self.args, 'val_interval', 0)
            if (val_interval > 0 and self.val_dataloader is not None
                    and self.n_steps % val_interval == 0):
                self.validation(max_val_batches=self.args.max_val_batches)
            # Debug only.
            # if i % 50 == 0:
            #     print(f"epoch: {epoch}, batch_id: {i}, av_loss: {av_loss}, bv_loss: {bv_loss}, "
            #           f"fg_loss: {fg_loss}, av_dist_loss: {av_dist_loss}, bv_dist_loss: {bv_dist_loss}, "
            #           f"fg_dist_loss: {fg_dist_loss}")

        cum_loss_sum /= cum_iter_count
        av_loss_sum /= cum_iter_count
        bv_loss_sum /= cum_iter_count
        fg_loss_sum /= cum_iter_count
        av_dist_loss_sum /= cum_iter_count
        bv_dist_loss_sum /= cum_iter_count
        fg_dist_loss_sum /= cum_iter_count

        val_loss = self.validation(max_val_batches=self.args.max_val_batches)

        return self.n_iter, cum_loss_sum, (av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum,
                                           bv_dist_loss_sum, fg_dist_loss_sum)

    def save(self, batch_idx, n_steps, epoch, file_path, name=None, save_last=False) -> str:
        """
        Save the intermediate models during training.
        :param batch_idx: the batch index
        :param n_steps: the step number
        :param epoch: the epoch number
        :param file_path: the file_path to save the model
        :param name: optional custom filename
        :param save_last: whether to save the last model
        :return: the output path
        """
        return save_checkpoint(self.model, self.optimizer, self.args, batch_idx, 
                               n_steps, epoch, file_path, name, save_last)


    def save_tmp(self, epoch, file_path, rank=0):
        """
        Save the models for auto-restore during training.
        The model are stored in file_path/tmp folder and will replaced on each epoch.
        :param epoch: the epoch number.
        :param file_path: the file_path to store the model.
        :param rank: the current rank (decrypted).
        :return:
        """
        store_path = os.path.join(file_path, "tmp")
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)
        store_path = os.path.join(store_path, "model.%d" % rank)
        state = {
            'args': self.args,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            "epoch": epoch
        }
        torch.save(state, store_path)

    def load(self, checkpoint_path):
        """
        Load checkpoint for training.
        :param checkpoint_path: path to checkpoint file
        :return: tuple of (epoch, scheduler_step, batch_idx, wandb_run_id)
        """
        epoch, scheduler_step, batch_idx, wandb_run_id = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.n_steps = scheduler_step
        return epoch, scheduler_step, batch_idx, wandb_run_id


class KERMTCMIMTrainer:
    """
    Trainer for CMIM pretraining.
    Uses contrastive learning without vocabulary prediction tasks.
    """
    def __init__(self,
                 args,
                 model: Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer,
                 scheduler,
                 gpu_id,
                 n_steps: int,
                 logger: Logger = None,
                 shutdown_checker=None):
        """
        The init function of KERMTCMIMTrainer
        :param args: the input arguments
        :param model: the complete KermtCMIMTask model (created externally)
        :param train_dataloader: the training dataloader
        :param val_dataloader: the validation dataloader
        :param optimizer: the optimizer (built on model.parameters())
        :param scheduler: the scheduler
        :param gpu_id: the gpu id
        :param n_steps: initial step count
        :param logger: the logger
        :param shutdown_checker: callable that returns True if graceful shutdown requested
        """
        self.args = args
        self.model = model
        self.loss_func = self.model.get_loss_func(args)
        self.gpu_id = gpu_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.debug = logger.debug if logger is not None else print
        self.shutdown_checker = shutdown_checker  # For graceful shutdown on cluster time limits

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_iter = 0

        self.model.to(self.gpu_id)
        # find_unused_parameters is required in two cases:
        #  * embedding_output_type != 'both': in atom/bond mode the encoder produces only
        #    one aggregation level, so the opposite branch's FFNs and the bond/atom vocab +
        #    FG heads receive no gradient.
        #  * a small --depth: the dyMPN depth draw can then land low enough to skip message
        #    passing entirely, leaving W_h without gradient on that batch. See
        #    dynamic_depth_can_skip_message_passing().
        # Neither holds for the usual 'both' + --depth 6, so the flag stays off there and
        # DDP keeps its fast path.
        # (static_graph is not an option: dynamic-depth sampling varies the graph per step.)
        self.model = DDP(self.model, device_ids=[gpu_id],
                         find_unused_parameters=(self.args.embedding_output_type != 'both'
                                                 or dynamic_depth_can_skip_message_passing(self.args.depth)))

        if self.args.tensorboard:
            self.writer = SummaryWriter(self.args.save_dir)

        self.n_steps = n_steps
        self.first_epoch_post_resume = True
        self.curr_epoch_batch_idx = 0
        self.batch_idx_offset = 0  # Offset to add when saving batch_idx (for sampler-level resume)

    def train(self, start_epoch: int, max_epochs: int) -> List:
        """
        The training iteration
        :param start_epoch: starting epoch number
        :param max_epochs: the max epochs
        :return: the loss terms of current epoch
        """
        for epoch in range(start_epoch, max_epochs):
            s_time = time.time()
            _, train_loss, _ = self.iter(epoch, train=True)
            t_time = time.time() - s_time
            if self.gpu_id == 0:
                print(f"epoch={epoch:04d}, cur_lr={self.scheduler.get_lr()[0]:.5f}, train_loss={train_loss:.6f}, train_time={t_time:.2f}", flush=True)
            # After the resumed epoch completes, reset sampler and offset so
            # subsequent epochs iterate over all samples from the beginning
            if epoch == start_epoch and hasattr(self.train_dataloader.sampler, 'set_start_index'):
                self.train_dataloader.sampler.set_start_index(0)
                self.batch_idx_offset = 0

    def validation(self, max_val_batches: int) -> float:
        """
        The validation iteration
        :param max_val_batches: the maximum number of batches to validate
        :return: the average validation loss
        """
        self.model.eval()
        loss_sum = 0
        n_batches = 0
        recon_loss_sum = 0
        cmim_loss_sum = 0
        log_p_k1_sum = 0
        log_q_z_sum = 0
        log_P_z_sum = 0
        recon_accuracy_sum = 0

        for ibatch, item in enumerate(self.val_dataloader):
            # Forward pass - pass whole batch to model
            preds = self.model(item)
            
            # CMIM+Reconstruction loss returns: (overall_loss, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, recon_accuracy)
            # Note: recon_accuracy will be None if args.tensorboard is False
            loss, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, recon_accuracy = self.loss_func(preds, targets=None)

            loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            cmim_loss_sum += cmim_loss.item()
            log_p_k1_sum += log_p_k1.item()
            log_q_z_sum += log_q_z.item()
            log_P_z_sum += log_P_z.item()
            if self.args.tensorboard:
                recon_accuracy_sum += recon_accuracy.item()

            n_batches += 1
            if n_batches >= max_val_batches:
                break

        # Compute per batch losses
        loss_sum /= n_batches
        recon_loss_sum /= n_batches
        cmim_loss_sum /= n_batches
        log_p_k1_sum /= n_batches
        log_q_z_sum /= n_batches
        log_P_z_sum /= n_batches
        if self.args.tensorboard:
            recon_accuracy_sum /= n_batches

        if self.gpu_id == 0:
            # Print with accuracy only if TensorBoard is enabled
            if self.args.tensorboard:
                print(f"Validation loss: {loss_sum:.4f}, recon_loss: {recon_loss_sum:.4f}, cmim_loss: {cmim_loss_sum:.4f}, "
                      f"log_p_k1: {log_p_k1_sum:.4f}, log_q_z: {log_q_z_sum:.4f}, log_P_z: {log_P_z_sum:.4f}, "
                      f"recon_acc: {recon_accuracy_sum:.4f}", flush=True)
            else:
                print(f"Validation loss: {loss_sum:.4f}, recon_loss: {recon_loss_sum:.4f}, cmim_loss: {cmim_loss_sum:.4f}, "
                      f"log_p_k1: {log_p_k1_sum:.4f}, log_q_z: {log_q_z_sum:.4f}, log_P_z: {log_P_z_sum:.4f}", flush=True)
            
            val_metrics = {
                'val/loss': loss_sum,
                'val/recon_loss': recon_loss_sum,
                'val/cmim_loss': cmim_loss_sum,
                'val/log_p_k1_given_zx': log_p_k1_sum,
                'val/log_q_z_given_x': log_q_z_sum,
                'val/log_P_z': log_P_z_sum,
                'val/recon_accuracy': recon_accuracy_sum,
            }
            if self.args.tensorboard:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(k, v, self.n_steps)
            _wandb_log(self.args, self.gpu_id, val_metrics, step=self.n_steps)

        self.model.train()
        return loss_sum

    def set_batch_idx(self, batch_idx: int, batch_idx_offset: int = 0):
        """
        Set batch index for resume logic.
        
        Args:
            batch_idx: Number of batches to skip in training loop (0 if sampler handles skipping)
            batch_idx_offset: Offset to add when saving batch_idx (for sampler-level resume)
        """
        self.curr_epoch_batch_idx = batch_idx
        self.batch_idx_offset = batch_idx_offset

    def iter(self, epoch, train=True) -> List:
        """
        Perform a training / validation iteration.
        :param epoch: the current epoch number
        :param train: True: train model, False: validation model
        :return: the loss terms as a list
        """
        if train:
            self.model.train()
            self.train_dataloader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        loss_sum, iter_count = 0, 0
        cum_loss_sum, cum_iter_count = 0, 0
        recon_loss_sum = 0
        cmim_loss_sum = 0
        log_p_k1_sum = 0
        log_q_z_sum = 0
        log_P_z_sum = 0
        recon_accuracy_sum = 0

        for ibatch, item in enumerate(self.train_dataloader):
            if self.first_epoch_post_resume:
                if ibatch < self.curr_epoch_batch_idx:
                    print(f"Skipping batch {ibatch} because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", flush=True)
                    continue
                elif ibatch >= self.curr_epoch_batch_idx:
                    print(f"Stop skipping batches because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", flush=True)
                    self.first_epoch_post_resume = False

            if self.gpu_id == 0:
                print(f"{self.n_steps=}", flush=True)
            
            # Forward pass - pass whole batch to model
            preds = self.model(item)

            # CMIM+Reconstruction loss returns: (overall_loss, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, recon_accuracy)
            # Note: recon_accuracy will be None if args.tensorboard is False
            loss, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, recon_accuracy = self.loss_func(preds, targets=None)

            loss_sum += loss.item()
            iter_count += self.args.batch_size

            if train:
                cum_loss_sum += loss.item()
                # Run model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                cum_loss_sum += loss.item()

            recon_loss_sum += recon_loss.item()
            cmim_loss_sum += cmim_loss.item()
            log_p_k1_sum += log_p_k1.item()
            log_q_z_sum += log_q_z.item()
            log_P_z_sum += log_P_z.item()
            if self.args.tensorboard:
                recon_accuracy_sum += recon_accuracy.item()

            # Save model (batch_idx includes offset for sampler-level resume)
            if (self.gpu_id == 0) and (self.n_steps % self.args.save_interval) == 0:
                self.save(batch_idx=ibatch + self.batch_idx_offset, n_steps=self.n_steps, epoch=epoch, 
                         file_path=self.args.save_dir, name=f"model_step_{self.n_steps}.pt", save_last=True)
                if self.args.tensorboard:
                    self.writer.flush()

            # Check for graceful shutdown (e.g., cluster time limit approaching)
            if self.shutdown_checker is not None and self.shutdown_checker():
                if self.gpu_id == 0:
                    # Save checkpoint before exiting
                    self.save(batch_idx=ibatch + self.batch_idx_offset, n_steps=self.n_steps, epoch=epoch, 
                              file_path=self.args.save_dir, name=f"model_step_{self.n_steps}_shutdown.pt", save_last=True)
                    if self.args.tensorboard:
                        self.writer.flush()
                        self.writer.close()
                    print(f"[SHUTDOWN] Graceful shutdown complete. Saved at step {self.n_steps}, batch {ibatch + self.batch_idx_offset}", flush=True)
                raise SystemExit(0)

            cum_iter_count += 1
            self.n_iter += self.args.batch_size
            self.n_steps += 1

            train_log_interval = max(1, self.args.train_interval)
            if self.gpu_id == 0 and self.n_steps % train_log_interval == 0:
                train_metrics = {
                    'train/loss': loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/cmim_loss': cmim_loss.item(),
                    'train/log_p_k1_given_zx': log_p_k1.item(),
                    'train/log_q_z_given_x': log_q_z.item(),
                    'train/log_P_z': log_P_z.item(),
                    'train/lr': self.scheduler.get_lr()[0],
                    'train/epoch': epoch,
                    'train/batch_idx': ibatch + self.batch_idx_offset,
                }
                # The loss function returns recon_accuracy=None unless --tensorboard is
                # set (see where it is unpacked above), so this must be guarded exactly
                # like the accumulation is. Same guard as KERMTHybridTrainer.
                if recon_accuracy is not None:
                    train_metrics['train/recon_accuracy'] = recon_accuracy.item()
                if self.args.tensorboard:
                    for k, v in train_metrics.items():
                        self.writer.add_scalar(k, v, self.n_steps)
                _wandb_log(self.args, self.gpu_id, train_metrics, step=self.n_steps)
            # Optional: run validation every val_interval steps (similar to train metrics / save_interval)
            val_interval = getattr(self.args, 'val_interval', 0)
            if (val_interval > 0 and self.val_dataloader is not None
                    and self.n_steps % val_interval == 0):
                self.validation(max_val_batches=self.args.max_val_batches)

        cum_loss_sum /= cum_iter_count
        cmim_loss_sum /= cum_iter_count
        log_p_k1_sum /= cum_iter_count
        log_q_z_sum /= cum_iter_count
        log_P_z_sum /= cum_iter_count
        if self.args.tensorboard:
            recon_accuracy_sum /= cum_iter_count

        val_loss = self.validation(max_val_batches=self.args.max_val_batches)

        return self.n_iter, cum_loss_sum, (cmim_loss_sum, log_p_k1_sum, log_q_z_sum, log_P_z_sum)

    def save(self, batch_idx, n_steps, epoch, file_path, name=None, save_last=False) -> str:
        """
        Save the intermediate models during training.
        :param batch_idx: the batch index
        :param n_steps: the step number
        :param epoch: the epoch number
        :param file_path: the file_path to save the model
        :param name: optional custom filename
        :param save_last: whether to save the last model
        :return: the output path
        """
        return save_checkpoint(self.model, self.optimizer, self.args, batch_idx, 
                              n_steps, epoch, file_path, name, save_last)

    def load(self, checkpoint_path):
        """
        Load checkpoint for CMIM training.
        :param checkpoint_path: path to checkpoint file
        :return: tuple of (epoch, scheduler_step, batch_idx, wandb_run_id)
        """
        epoch, scheduler_step, batch_idx, wandb_run_id = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.n_steps = scheduler_step
        return epoch, scheduler_step, batch_idx, wandb_run_id


class KERMTHybridTrainer:
    """
    Trainer for hybrid CMIM + vocabulary pretraining.
    Combines contrastive learning, SMILES reconstruction, and vocab prediction objectives.
    """
    def __init__(self,
                 args,
                 model: Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer,
                 scheduler,
                 gpu_id,
                 n_steps: int,
                 logger: Logger = None,
                 shutdown_checker=None):
        """
        The init function of KERMTHybridTrainer.
        
        :param args: the input arguments
        :param model: the complete KermtHybridTask model
        :param train_dataloader: the training dataloader
        :param val_dataloader: the validation dataloader
        :param optimizer: the optimizer
        :param scheduler: the scheduler
        :param gpu_id: the gpu id
        :param n_steps: initial step count
        :param logger: the logger
        :param shutdown_checker: callable that returns True if graceful shutdown requested
        """
        self.args = args
        self.model = model
        self.loss_func = self.model.get_loss_func(args)
        self.gpu_id = gpu_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.debug = logger.debug if logger is not None else print
        self.shutdown_checker = shutdown_checker  # For graceful shutdown on cluster time limits

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_iter = 0

        self.model.to(self.gpu_id)
        # find_unused_parameters is required in two cases:
        #  * embedding_output_type != 'both': in atom/bond mode the encoder produces only
        #    one aggregation level, so the opposite branch's FFNs and the bond/atom vocab +
        #    FG heads receive no gradient.
        #  * a small --depth: the dyMPN depth draw can then land low enough to skip message
        #    passing entirely, leaving W_h without gradient on that batch. See
        #    dynamic_depth_can_skip_message_passing().
        # Neither holds for the usual 'both' + --depth 6, so the flag stays off there and
        # DDP keeps its fast path.
        # (static_graph is not an option: dynamic-depth sampling varies the graph per step.)
        self.model = DDP(self.model, device_ids=[gpu_id],
                         find_unused_parameters=(self.args.embedding_output_type != 'both'
                                                 or dynamic_depth_can_skip_message_passing(self.args.depth)))

        if self.args.tensorboard:
            self.writer = SummaryWriter(self.args.save_dir)

        self.n_steps = n_steps
        self.first_epoch_post_resume = True
        self.curr_epoch_batch_idx = 0
        self.batch_idx_offset = 0  # Offset to add when saving batch_idx (for sampler-level resume)

    def train(self, start_epoch: int, max_epochs: int) -> List:
        """
        The training iteration.
        :param start_epoch: starting epoch number
        :param max_epochs: the max epochs
        :return: the loss terms of current epoch
        """
        for epoch in range(start_epoch, max_epochs):
            s_time = time.time()
            _, train_loss, _ = self.iter(epoch, train=True)
            t_time = time.time() - s_time
            if self.gpu_id == 0:
                print(f"epoch={epoch:04d}, cur_lr={self.scheduler.get_lr()[0]:.5f}, "
                      f"train_loss={train_loss:.6f}, train_time={t_time:.2f}", flush=True)
            # After the resumed epoch completes, reset sampler and offset so
            # subsequent epochs iterate over all samples from the beginning
            if epoch == start_epoch and hasattr(self.train_dataloader.sampler, 'set_start_index'):
                self.train_dataloader.sampler.set_start_index(0)
                self.batch_idx_offset = 0

    def validation(self, max_val_batches: int) -> float:
        """
        The validation iteration.
        :param max_val_batches: the maximum number of batches to validate
        :return: the average validation loss
        """
        self.model.eval()
        n_batches = 0
        
        # Initialize accumulators for all loss components
        loss_sum = 0
        cmim_total_sum = 0
        recon_loss_sum = 0
        cmim_loss_sum = 0
        log_p_k1_sum = 0
        log_q_z_sum = 0
        log_P_z_sum = 0
        recon_accuracy_sum = 0
        vocab_loss_sum = 0
        av_loss_sum = 0
        bv_loss_sum = 0
        fg_loss_sum = 0

        for ibatch, item in enumerate(self.val_dataloader):
            # Move vocab targets to GPU
            targets = item["targets"]
            targets["av_task"] = targets["av_task"].to(self.gpu_id)
            targets["bv_task"] = targets["bv_task"].to(self.gpu_id)
            targets["fg_task"] = targets["fg_task"].to(self.gpu_id)
            
            # Forward pass
            preds = self.model(item)
            
            # Compute loss - returns all components
            (overall_loss, cmim_total, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, 
             recon_accuracy, vocab_overall, av_loss, bv_loss, fg_loss, 
             av_dist_loss, bv_dist_loss, fg_dist_loss) = self.loss_func(preds, targets)

            loss_sum += overall_loss.item()
            cmim_total_sum += cmim_total.item()
            recon_loss_sum += recon_loss.item()
            cmim_loss_sum += cmim_loss.item()
            log_p_k1_sum += log_p_k1.item()
            log_q_z_sum += log_q_z.item()
            log_P_z_sum += log_P_z.item()
            if recon_accuracy is not None:
                recon_accuracy_sum += recon_accuracy.item()
            vocab_loss_sum += vocab_overall.item() if not isinstance(vocab_overall, float) else vocab_overall
            av_loss_sum += av_loss.item() if not isinstance(av_loss, float) else av_loss
            bv_loss_sum += bv_loss.item() if not isinstance(bv_loss, float) else bv_loss
            fg_loss_sum += fg_loss.item() if not isinstance(fg_loss, float) else fg_loss

            n_batches += 1
            if n_batches >= max_val_batches:
                break

        # Compute per batch averages
        loss_sum /= n_batches
        cmim_total_sum /= n_batches
        recon_loss_sum /= n_batches
        cmim_loss_sum /= n_batches
        log_p_k1_sum /= n_batches
        log_q_z_sum /= n_batches
        log_P_z_sum /= n_batches
        recon_accuracy_sum /= n_batches
        vocab_loss_sum /= n_batches
        av_loss_sum /= n_batches
        bv_loss_sum /= n_batches
        fg_loss_sum /= n_batches

        if self.gpu_id == 0:
            print(f"Validation loss: {loss_sum:.4f}, cmim_total: {cmim_total_sum:.4f}, "
                  f"vocab_loss: {vocab_loss_sum:.4f}, recon_loss: {recon_loss_sum:.4f}", flush=True)
            
            val_metrics = {
                'val/loss': loss_sum,
                'val/cmim_total': cmim_total_sum,
                'val/recon_loss': recon_loss_sum,
                'val/cmim_loss': cmim_loss_sum,
                'val/log_p_k1_given_zx': log_p_k1_sum,
                'val/log_q_z_given_x': log_q_z_sum,
                'val/log_P_z': log_P_z_sum,
                'val/recon_accuracy': recon_accuracy_sum,
                'val/vocab_loss': vocab_loss_sum,
                'val/av_loss': av_loss_sum,
                'val/bv_loss': bv_loss_sum,
                'val/fg_loss': fg_loss_sum,
            }
            if self.args.tensorboard:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(k, v, self.n_steps)
            _wandb_log(self.args, self.gpu_id, val_metrics, step=self.n_steps)

        self.model.train()
        return loss_sum

    def set_batch_idx(self, batch_idx: int, batch_idx_offset: int = 0):
        """
        Set batch index for resume logic.
        
        Args:
            batch_idx: Number of batches to skip in training loop (0 if sampler handles skipping)
            batch_idx_offset: Offset to add when saving batch_idx (for sampler-level resume)
        """
        self.curr_epoch_batch_idx = batch_idx
        self.batch_idx_offset = batch_idx_offset

    def iter(self, epoch, train=True) -> List:
        """
        Perform a training / validation iteration.
        :param epoch: the current epoch number
        :param train: True: train model, False: validation model
        :return: the loss terms as a list
        """
        if train:
            self.model.train()
            self.train_dataloader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        loss_sum, iter_count = 0, 0
        cum_loss_sum, cum_iter_count = 0, 0
        
        # Accumulators for logging
        cmim_total_sum = 0
        recon_loss_sum = 0
        cmim_loss_sum = 0
        log_p_k1_sum = 0
        log_q_z_sum = 0
        log_P_z_sum = 0
        recon_accuracy_sum = 0
        vocab_loss_sum = 0
        av_loss_sum = 0
        bv_loss_sum = 0
        fg_loss_sum = 0

        for ibatch, item in enumerate(self.train_dataloader):
            if self.first_epoch_post_resume:
                if ibatch < self.curr_epoch_batch_idx:
                    print(f"Skipping batch {ibatch} because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", 
                          flush=True)
                    continue
                elif ibatch >= self.curr_epoch_batch_idx:
                    print(f"Stop skipping batches because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", 
                          flush=True)
                    self.first_epoch_post_resume = False

            if self.gpu_id == 0:
                print(f"{self.n_steps=}", flush=True)
            
            # Move vocab targets to GPU
            targets = item["targets"]
            targets["av_task"] = targets["av_task"].to(self.gpu_id)
            targets["bv_task"] = targets["bv_task"].to(self.gpu_id)
            targets["fg_task"] = targets["fg_task"].to(self.gpu_id)
            
            # Forward pass
            preds = self.model(item)

            # Compute loss - returns all components
            (overall_loss, cmim_total, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, 
             recon_accuracy, vocab_overall, av_loss, bv_loss, fg_loss, 
             av_dist_loss, bv_dist_loss, fg_dist_loss) = self.loss_func(preds, targets)

            loss_sum += overall_loss.item()
            iter_count += self.args.batch_size

            if train:
                cum_loss_sum += overall_loss.item()
                self.optimizer.zero_grad()
                overall_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                cum_loss_sum += overall_loss.item()

            # Accumulate for logging
            cmim_total_sum += cmim_total.item()
            recon_loss_sum += recon_loss.item()
            cmim_loss_sum += cmim_loss.item()
            log_p_k1_sum += log_p_k1.item()
            log_q_z_sum += log_q_z.item()
            log_P_z_sum += log_P_z.item()
            if recon_accuracy is not None:
                recon_accuracy_sum += recon_accuracy.item()
            vocab_loss_sum += vocab_overall.item() if not isinstance(vocab_overall, float) else vocab_overall
            av_loss_sum += av_loss.item() if not isinstance(av_loss, float) else av_loss
            bv_loss_sum += bv_loss.item() if not isinstance(bv_loss, float) else bv_loss
            fg_loss_sum += fg_loss.item() if not isinstance(fg_loss, float) else fg_loss

            # Save model (batch_idx includes offset for sampler-level resume)
            if (self.gpu_id == 0) and (self.n_steps % self.args.save_interval) == 0:
                self.save(batch_idx=ibatch + self.batch_idx_offset, n_steps=self.n_steps, epoch=epoch,
                         file_path=self.args.save_dir, name=f"model_step_{self.n_steps}.pt", save_last=True)
                if self.args.tensorboard:
                    self.writer.flush()

            # Check for graceful shutdown (e.g., cluster time limit approaching)
            if self.shutdown_checker is not None and self.shutdown_checker():
                if self.gpu_id == 0:
                    # Save checkpoint before exiting
                    self.save(batch_idx=ibatch + self.batch_idx_offset, n_steps=self.n_steps, epoch=epoch,
                              file_path=self.args.save_dir, name=f"model_step_{self.n_steps}_shutdown.pt", save_last=True)
                    if self.args.tensorboard:
                        self.writer.flush()
                        self.writer.close()
                    print(f"[SHUTDOWN] Graceful shutdown complete. Saved at step {self.n_steps}, batch {ibatch + self.batch_idx_offset}", flush=True)
                raise SystemExit(0)

            cum_iter_count += 1
            self.n_iter += self.args.batch_size
            self.n_steps += 1

            train_log_interval = max(1, self.args.train_interval)
            if self.gpu_id == 0 and self.n_steps % train_log_interval == 0:
                vocab_overall_val = vocab_overall.item() if not isinstance(vocab_overall, float) else vocab_overall
                av_loss_val = av_loss.item() if not isinstance(av_loss, float) else av_loss
                bv_loss_val = bv_loss.item() if not isinstance(bv_loss, float) else bv_loss
                fg_loss_val = fg_loss.item() if not isinstance(fg_loss, float) else fg_loss
                train_metrics = {
                    'train/loss': overall_loss.item(),
                    'train/cmim_total': cmim_total.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/cmim_loss': cmim_loss.item(),
                    'train/log_p_k1_given_zx': log_p_k1.item(),
                    'train/log_q_z_given_x': log_q_z.item(),
                    'train/log_P_z': log_P_z.item(),
                    'train/vocab_loss': vocab_overall_val,
                    'train/av_loss': av_loss_val,
                    'train/bv_loss': bv_loss_val,
                    'train/fg_loss': fg_loss_val,
                    'train/lr': self.scheduler.get_lr()[0],
                    'train/epoch': epoch,
                    'train/batch_idx': ibatch + self.batch_idx_offset,
                }
                if recon_accuracy is not None:
                    train_metrics['train/recon_accuracy'] = recon_accuracy.item()
                if self.args.tensorboard:
                    for k, v in train_metrics.items():
                        self.writer.add_scalar(k, v, self.n_steps)
                _wandb_log(self.args, self.gpu_id, train_metrics, step=self.n_steps)
            # Optional: run validation every val_interval steps (similar to train metrics / save_interval)
            val_interval = getattr(self.args, 'val_interval', 0)
            if (val_interval > 0 and self.val_dataloader is not None
                    and self.n_steps % val_interval == 0):
                self.validation(max_val_batches=self.args.max_val_batches)

        cum_loss_sum /= cum_iter_count
        cmim_loss_sum /= cum_iter_count
        vocab_loss_sum /= cum_iter_count

        val_loss = self.validation(max_val_batches=self.args.max_val_batches)

        return self.n_iter, cum_loss_sum, (cmim_loss_sum, vocab_loss_sum)

    def save(self, batch_idx, n_steps, epoch, file_path, name=None, save_last=False) -> str:
        """
        Save the intermediate models during training.
        """
        return save_checkpoint(self.model, self.optimizer, self.args, batch_idx,
                              n_steps, epoch, file_path, name, save_last)

    def load(self, checkpoint_path):
        """
        Load checkpoint for hybrid training.
        :param checkpoint_path: path to checkpoint file
        :return: tuple of (epoch, scheduler_step, batch_idx, wandb_run_id)
        """
        epoch, scheduler_step, batch_idx, wandb_run_id = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.n_steps = scheduler_step
        return epoch, scheduler_step, batch_idx, wandb_run_id
