#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evidence accumulation task (modeled on https://www.nature.com/articles/s41467-019-11050-x; 
                            see Fig 5 in paper)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def sample_data(
    batch_size,
    seq_len,
    puff_density=0.5,
    noise_var=0.,
    ncheeks=2,
    rng=None,
    delay=0,
):
            
    inputs = np.zeros((batch_size*seq_len, ncheeks))    
    targets = np.zeros(batch_size)
    
    #one cheek or the other...
    on_cheek = rng.randint(0, ncheeks, size=(batch_size*seq_len))
    
    inputs[np.arange(len(inputs)), on_cheek] = 1
    
    #sparsify input
    sparse_mask = rng.rand(batch_size*seq_len) > puff_density
    inputs[sparse_mask] = 0    
      
    #turn into final shape
    inputs = inputs.reshape(batch_size, seq_len, ncheeks)
    
    #add some delay if required
    if delay > 0:
        inputs[:, seq_len - delay:, :] = 0
    
    
    npuffs_bias = np.sum(inputs, axis=1)
    
    npuffs_argmax = np.argmax(npuffs_bias, axis=1)
    
    targets = npuffs_argmax + 1 #leave 0 for draws...
    
    npuffs_max = np.max(npuffs_bias, axis=1)
    
    is_max = npuffs_bias == npuffs_max[:, None].repeat(2, axis=1)
    two_solutions = np.sum(is_max, axis=1) > 1
    
    targets[two_solutions] = 0
    
    if noise_var > 0:
        inputs = inputs + rng.randn(batch_size, seq_len, ncheeks)*noise_var
    
    return inputs, targets


class BatchGenerator:
    def __init__(
        self, size=1000, seq_len=10, noise_var=0.0,
        batch_size=10, offline_data=False,
        random_state=None, ncheeks=2, puff_density=0.5,
        delay=0
    ):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.size = size
        self.noise_var = noise_var
        self.offline_data = offline_data
        
        self.ncheeks = ncheeks
        self.puff_density = puff_density
        self.delay = delay

        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        self.rng = random_state
        self.init_state = random_state.get_state()

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

        if self.offline_data:
            self.reset()

    def __len__(self):
        return int(np.ceil(self.size / self.batch_size))

    def n_instances(self):
        return self.size * self.batch_size

    def reset(self):
        self.rng.set_state(self.init_state)

    def next_batch(self):
        inputs, targets = sample_data(
            batch_size=self.batch_size, seq_len=self.seq_len,
            noise_var=self.noise_var, rng=self.rng,
            ncheeks=self.ncheeks, puff_density=self.puff_density,
            delay=self.delay
        )

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        targets = torch.as_tensor(targets, dtype=torch.long)

        return inputs, targets


    def torch_dataset(self):
        current_state = self.rng.get_state()
        self.rng.set_state(self.init_state)

        inputs, targets = zip(*[batch for batch in self])

        self.rng.set_state(current_state)

        data = TensorDataset(torch.cat(inputs), torch.cat(targets))

        return DataLoader(
            dataset=data,
            batch_size=self.batch_size, shuffle=True
        )


def load_evidence_accumulation(
    training_size,
    test_size,
    batch_size,
    seq_len,
    train_val_split,
    train_noise_var,
    test_noise_var,
    ncheeks=2,
    puff_density=0.5,
    fixdata=False,
    random_state=None,
    delay=0
):
    
    N = int(training_size * (1 - train_val_split))
    val_size = training_size - N

    if random_state is None:
        train_rng = np.random.randint(2**16-1)
        val_rng = np.random.randint(2**16-1)
        test_rng = np.random.randint(2**16-1)
    else:
        train_rng = random_state.randint(2**16-1)
        val_rng = random_state.randint(2**16-1)
        test_rng = random_state.randint(2**16-1)

    training_data = BatchGenerator(
        size=N, seq_len=seq_len, batch_size=batch_size,
        noise_var=train_noise_var, offline_data=fixdata, random_state=train_rng, 
        ncheeks=ncheeks, puff_density=puff_density, delay=delay)

    validation_data = BatchGenerator(
        size=val_size, seq_len=seq_len, batch_size=val_size,
        noise_var=train_noise_var, offline_data=fixdata, random_state=val_rng, 
        ncheeks=ncheeks, puff_density=puff_density, delay=delay)

    test_data = BatchGenerator(
        size=test_size, seq_len=seq_len, batch_size=test_size,
        noise_var=test_noise_var, offline_data=fixdata, random_state=test_rng,
        ncheeks=ncheeks, puff_density=puff_density, delay=delay)

    if fixdata:
        training_data  = training_data.torch_dataset()
        validation_data = validation_data.torch_dataset()
        test_data = test_data.torch_dataset()

    return training_data, validation_data, test_data
