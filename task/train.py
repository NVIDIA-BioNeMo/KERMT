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
The training function used in the finetuning task.
"""
import csv
import logging
import os
import json
import pickle  # noqa: F401  (kept; available if a caller wants to load legacy .pckl splits)
import time
from argparse import Namespace
from logging import Logger
from typing import List

import numpy as np
import pandas as pd
try:
    import wandb
except ImportError:
    wandb = None
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kermt.data import MolCollator
from kermt.data import StandardScaler
from kermt.util.loss import MTLLoss
from kermt.util.metrics import get_metric_func
from kermt.util.nn_utils import initialize_weights, param_count_trainable, param_count_total
from kermt.util.scheduler import NoamLR
from kermt.util.utils import build_optimizer, build_lr_scheduler, makedirs, load_checkpoint, get_loss_func, \
    save_checkpoint, build_model, get_ffn_layer_names, save_model_for_restart
from kermt.util.utils import get_class_sizes, get_data, split_data, get_task_names
from task.predict import predict, evaluate, evaluate_predictions



def train(epoch, model, data, loss_func, mtl_loss, optimizer, scheduler,
          shared_dict, args: Namespace, n_iter: int = 0,
          logger: logging.Logger = None):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :return: The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print
    model.train()

    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0


    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    num_workers = 0
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model. optimizer.zero_grad() clears the gradient of every
        # registered parameter, including mtl_loss.log_sigma (appended to the
        # optimizer's param groups when --use_mtl_loss). model.zero_grad() alone
        # would miss it, letting the MTL task-weight gradient accumulate.
        optimizer.zero_grad()
        preds = model(batch, features_batch)
        loss = loss_func(preds, targets) * class_weights * mask

        if mtl_loss is not None:
            # Compute per-task mean losses, handling division by zero for tasks with no valid samples
            task_mask_sum = mask.sum(axis=0)
            task_mask_sum = torch.clamp(task_mask_sum, min=1.0)  # Avoid division by zero
            task_losses = loss.sum(axis=0) / task_mask_sum
            loss = mtl_loss(task_losses)
        else:
            loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

        #if (n_iter // args.batch_size) % args.log_frequency == 0:
        #    lrs = scheduler.get_lr()
        #    loss_avg = loss_sum / iter_count
        #    loss_sum, iter_count = 0, 0
        #    lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
        # if cum_iter_count % 10 == 0:
        #     current_mem, peak_mem = get_memory_usage()
        #     print(f"After {cum_iter_count} iterations: {current_mem}MB, {peak_mem}MB")

    return n_iter, cum_loss_sum / cum_iter_count


def run_training(args: Namespace, logger: Logger = None, return_val=False) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print


    # pin GPU to local rank.
    idx = args.gpu
    if args.gpu is not None:
        torch.cuda.set_device(idx)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    
    # Check if test data is blinded (no target columns)
    is_blinded_test = len(test_targets) == 0 or (len(test_targets) > 0 and len(test_targets[0]) == 0)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        
        # Initialize TensorBoard writer if enabled
        if args.tensorboard:
            writer = SummaryWriter(save_dir)

        # Load/build model
        start_epoch = 0  # Default: start from epoch 0
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model, loaded_ckpt_state = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)
            loaded_ckpt_state = {}

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        if args.use_mtl_loss:
            mtl_loss = MTLLoss(args.num_tasks)
        else:
            mtl_loss = None

        debug(model)
        debug(f'Number of trainable parameters = {param_count_trainable(model):,}')
        debug(f'Number of total parameters = {param_count_total(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()
            if mtl_loss is not None:
                mtl_loss = mtl_loss.cuda()
        optimizer = build_optimizer(model, args)
        if args.use_mtl_loss:
            # Train log_sigma with same LR as task head (FFN params), not encoder
            optimizer.param_groups[1]['params'].append(mtl_loss.log_sigma)

        # Try to load optimizer state - only use start_epoch if optimizer loads successfully
        # (indicates resuming a finetune job vs starting fresh from pretrain checkpoint)
        if args.checkpoint_paths is not None and "optimizer" in loaded_ckpt_state:
            try:
                print(f"Loading optimizer state from checkpoint: {args.checkpoint_paths[cur_model]}")
                optimizer.load_state_dict(loaded_ckpt_state["optimizer"])
                # Only resume from checkpoint epoch if optimizer loaded successfully
                if "epoch" in loaded_ckpt_state:
                    start_epoch = loaded_ckpt_state["epoch"]
                    print(f"Resuming from epoch {start_epoch}")
            except ValueError as e:
                print(f"Could not load optimizer state (model structure may differ): {e}")
                print("Starting fresh finetuning from epoch 0.")

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)
        # Only load scheduler state if we're resuming (start_epoch > 0 means optimizer loaded successfully)
        if start_epoch > 0 and "scheduler" in loaded_ckpt_state:
            try:
                print(f"Loading scheduler state from checkpoint: {args.checkpoint_paths[cur_model]}")
                scheduler.load_state_dict(loaded_ckpt_state["scheduler"])
            except (ValueError, KeyError) as e:
                print(f"Could not load scheduler state: {e}")
                print("Starting with fresh scheduler state.")

        # Bulid data_loader
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        # Use a separate name so train_data stays the underlying dataset across
        # ensemble iterations; reassigning it would double-wrap the DataLoader
        # (DataLoader(DataLoader(...))) on the second and later ensemble models.
        train_loader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=shuffle,
                                  num_workers=0,
                                  collate_fn=mol_collator)
        # Run training
        if args.task_wise_checkpoint:
            best_score = {task: float('inf') if args.minimize_score else -float('inf') for task in args.task_names}
            curr_epoch_best_by_loss = {}
        else:
            best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch = {task: 0 for task in args.task_names}
        n_iter = 0

        # Initialize validation losses
        if args.task_wise_checkpoint:
            min_val_loss = {task: float('inf') for task in args.task_names}
        else:
            min_val_loss = float('inf')
        for epoch in range(start_epoch, args.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                epoch=epoch,
                model=model,
                data=train_loader,
                loss_func=loss_func,
                mtl_loss=mtl_loss,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger
            )
            t_time = time.time() - s_time
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            avg_val_loss = np.nanmean(val_loss)
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(avg_val_loss),
                  f'{args.metric}_val: {avg_val_score:.4f}',
                  'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time),
                  flush=True)
            if args.wandb_project:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/time": t_time,
                    "val/time": v_time,
                    "val/loss": avg_val_loss,
                    f"val/{args.metric}": avg_val_score,
                    "cur_lr": scheduler.get_lr()[-1],
                }
                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, val_scores):
                        log_dict[f"val/{task_name}_{args.metric}"] = val_score
                wandb.log(log_dict)

            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', avg_val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)

            # Always update min_val_loss as it is needed for HPO
            if args.task_wise_checkpoint:

                for itask, task_name in enumerate(args.task_names):
                    if val_loss[itask] < min_val_loss[task_name]:
                        curr_epoch_best_by_loss[task_name] = True
                        min_val_loss[task_name], best_epoch[task_name] = val_loss[itask], epoch
                    else:
                        curr_epoch_best_by_loss[task_name] = False
            else:
                if avg_val_loss < min_val_loss:
                    curr_epoch_best_by_loss = True
                    min_val_loss, best_epoch = avg_val_loss, epoch
                else:
                    curr_epoch_best_by_loss = False

            save_model_for_restart(os.path.join(save_dir, 'last_checkpoint.pt'), model, optimizer, scheduler, scaler, features_scaler, args, 
            epoch+1 # save with +1 so that loaded checkpoint will start from the next epoch
            )
            # Save model checkpoint if improved validation score
            if args.task_wise_checkpoint:
                if args.select_by_loss:
                    for task_name in args.task_names:
                        if curr_epoch_best_by_loss[task_name]:
                            print(f"Saving model {task_name} at epoch {epoch} with validation loss {min_val_loss[task_name]:.4f}")
                            save_checkpoint(os.path.join(save_dir, f'model_{task_name}.pt'), model, scaler, features_scaler, args)
                else:
                    for itask, task_name in enumerate(args.task_names):
                        task_val_score = val_scores[itask] if itask < len(val_scores) else avg_val_score
                        if args.minimize_score and task_val_score < best_score[task_name] or \
                                not args.minimize_score and task_val_score > best_score[task_name]:
                            best_score[task_name], best_epoch[task_name] = task_val_score, epoch
                            print(f"Saving model {task_name} at epoch {epoch} with validation score {best_score[task_name]:.4f}")
                            save_checkpoint(os.path.join(save_dir, f'model_{task_name}.pt'), model, scaler, features_scaler, args)
            else:
                if args.select_by_loss:
                    if curr_epoch_best_by_loss:
                        print(f"Saving model at epoch {epoch} with validation loss {min_val_loss:.4f}")
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                else:
                    if args.minimize_score and avg_val_score < best_score or \
                            not args.minimize_score and avg_val_score > best_score:
                        best_score, best_epoch = avg_val_score, epoch
                        print(f"Saving model at epoch {epoch} with validation score {best_score:.4f}")
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            # TODO: Reimplement this
            # if epoch - best_epoch > args.early_stop_epoch:
            #     break

        ensemble_scores = 0.0

        # Evaluate on test set using model with best validation score
        if args.select_by_loss:
            if args.task_wise_checkpoint:
                for task_name in args.task_names:
                    info(f'Model {model_idx} best val loss = {min_val_loss[task_name]:.6f} on epoch {best_epoch[task_name]}')
            else:
                info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            if args.task_wise_checkpoint:
                for task_name in args.task_names:
                    info(f'Model {model_idx} best validation {args.metric} = {best_score[task_name]:.6f} on epoch {best_epoch[task_name]}')
            else:
                info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

        if args.task_wise_checkpoint:
            test_preds = np.zeros((len(test_data), args.num_tasks))
            test_scores = []
            _loss_func = None if is_blinded_test else loss_func
            for itask, task_name in enumerate(args.task_names):
                print(f"{itask=}, {task_name=}")
                task_model, _ = load_checkpoint(os.path.join(save_dir, f'model_{task_name}.pt'), cuda=args.cuda, logger=logger)
                test_preds_task, _ = predict(
                    model=task_model,
                    data=test_data,
                    loss_func=_loss_func,
                    batch_size=args.batch_size,
                    logger=logger,
                    shared_dict=shared_dict,
                    scaler=scaler,
                    args=args
                )
                if not is_blinded_test:
                    test_scores_task = evaluate_predictions(
                        preds=test_preds_task,
                        targets=test_targets,
                        num_tasks=args.num_tasks,
                        metric_func=metric_func,
                        dataset_type=args.dataset_type,
                        logger=logger
                    )
                    test_scores.append(test_scores_task[itask])
                test_preds[:, itask] = np.array(test_preds_task)[:, itask]
            if is_blinded_test:
                test_scores = [float('nan')] * args.num_tasks
        else:
            model, _ = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
            test_preds, _ = predict(
                model=model,
                data=test_data,
                loss_func=None if is_blinded_test else loss_func,
                batch_size=args.batch_size,
                logger=logger,
                shared_dict=shared_dict,
                scaler=scaler,
                args=args
            )
            if not is_blinded_test:
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    dataset_type=args.dataset_type,
                    logger=logger
                )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds, dtype=float)

        if not is_blinded_test and not args.task_wise_checkpoint:
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

        if not is_blinded_test:
            # Average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
        else:
            info(f'Model {model_idx} test: Blinded data - skipping metric evaluation')

        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        if not is_blinded_test:
            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Output with both predictions and targets
            ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
            ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
            data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
            test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

            # Average ensemble score
            avg_ensemble_test_score = np.nanmean(ensemble_scores)
            info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

            # Individual ensemble scores
            if args.show_individual_scores:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
            if args.wandb_project:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    wandb.summary[f"test/{task_name}_{args.metric}"] = ensemble_score
        else:
            # Blinded test data - output predictions only
            ensemble_scores = [float('nan')] * args.num_tasks
            test_result = pd.DataFrame(avg_test_preds, index=test_smiles, columns=args.task_names)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))
            info(f'Ensemble test: Blinded data - predictions saved to test_result.csv')
        
        # Close TensorBoard writer
        if args.tensorboard:
            writer.close()

    if return_val:
        return ensemble_scores, min_val_loss
    else:
        return ensemble_scores


def load_data(args, debug, logger):
    """
    load the training data.
    :param args:
    :param debug:
    :param logger:
    :return:
    """
    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)
    else:
        args.features_dim = 0
    shared_dict = {}
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    #if args.save_smiles_splits:
    #    save_splits(args, test_data, train_data, val_data)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)
    else:
        scaler = None
    return features_scaler, scaler, shared_dict, test_data, train_data, val_data


def save_splits(args, test_data, train_data, val_data):
    """
    Save the splits.
    :param args:
    :param test_data:
    :param train_data:
    :param val_data:
    :return:
    """
    with open(args.data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        lines_by_smiles = {}
        indices_by_smiles = {}
        for i, line in enumerate(reader):
            smiles = line[0]
            lines_by_smiles[smiles] = line
            indices_by_smiles[smiles] = i

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            for smiles in dataset.smiles():
                writer.writerow([smiles])
        with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in dataset.smiles():
                writer.writerow(lines_by_smiles[smiles])
        split_indices = []
        for smiles in dataset.smiles():
            split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
        all_split_indices.append(split_indices)
    with open(os.path.join(args.save_dir, 'split_indices.json'), 'w') as f:
        json.dump(all_split_indices, f)
    return writer
