#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
digit-drawing task (see Fig 4 in paper)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .draw_numbers import draw_number
except:
    from draw_numbers import draw_number


def define_inputs(input_D, min_value, max_value, npoints):
    nvalues = max_value - min_value + 1
    nposs_points = nvalues**input_D
    
    if input_D > 1:
        assert nposs_points >= npoints, 'Not enough possible inputs for required number of points'
    
    
    if input_D == 1:
        inputs = np.arange(npoints, dtype=float)[:, None]
    else:       
        inputs = np.zeros((npoints, input_D))
        
        #fix random state for consistency
        rs = np.random.RandomState(123)
        
        rand_numbers = rs.choice(nposs_points, size=npoints, replace=False)
        
        for i, number in enumerate(rand_numbers):
            bin_string = bin(number)[2:]
            bin_array = [int(x) for x in bin_string]
            inputs[i, -len(bin_array):] = bin_array
            
    return inputs

def define_targets(npoints, seqlen):
    assert npoints <= 10, 'Only 10 MNIST digits available!'
    
    targets = np.zeros((npoints, seqlen, 2))   
    
    for i in range(npoints):
        targets[i] = draw_number(i, seqlen)
    
    return targets
    

def get_input_target_pairs(input_D, seqlen, min_value, max_value, npoints):
    inputs = define_inputs(input_D, min_value, max_value, npoints)
    targets = define_targets(npoints, seqlen)
        
    return inputs, targets

def sample_data(
    batch_size,
    seq_len,
    input_vals,
    target_vals,
    noise_var,
    rng=None, 
):
    
    inds = rng.randint(0, len(input_vals), batch_size)
    
    inputs = np.zeros((batch_size, seq_len, input_vals.shape[1]))
    
    inputs[:, 0] = input_vals[inds]
    targets = target_vals[inds]

    if noise_var > 0:        
        inputs = inputs[:, :, :].astype(float) + rng.randn(inputs.shape[0], inputs.shape[1], inputs.shape[2])*noise_var
    return inputs, targets

class BatchGenerator:
    def __init__(
        self, input_vals, target_vals, size=1000, batch_size=1000, seq_len=10, 
        noise_var=0.0, offline_data=False,
        random_state=None, npoints = 1,
    ):
        self.size = size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.noise_var = noise_var
        self.offline_data = offline_data
        self.input_vals = input_vals
        self.target_vals = target_vals

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
            input_vals=self.input_vals, target_vals=self.target_vals,
            rng=self.rng, noise_var=self.noise_var      
        )

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        targets = torch.as_tensor(targets, dtype=torch.float)

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

def load_digit_draw(
    training_size,
    test_size,
    batch_size,
    seq_len,
    train_val_split=0.2,
    train_noise_var=0,
    test_noise_var=0,
    min_value=0,
    max_value=1,
    fixdata=False,
    random_state=None,
    input_D=1,
    npoints=6, 
):    
    
    if input_D > 1:
        assert min_value == 0, 'not ready for non-zero min value'
        assert max_value == 1, 'not ready for non-one max value'
                
    input_vals, target_vals = get_input_target_pairs(input_D, seq_len, min_value, max_value, npoints,)
    
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
        input_vals=input_vals, target_vals=target_vals,
        size=training_size, batch_size=batch_size, seq_len=seq_len, 
        noise_var=train_noise_var, offline_data=fixdata, random_state=train_rng, 
        )

    validation_data = BatchGenerator(
        input_vals=input_vals, target_vals=target_vals,
        size=val_size, batch_size=val_size, seq_len=seq_len,
        noise_var=train_noise_var, offline_data=fixdata, random_state=val_rng, 
        )

    test_data = BatchGenerator(
        input_vals=input_vals, target_vals=target_vals,
        size=test_size, batch_size=test_size, seq_len=seq_len,
        noise_var=test_noise_var, offline_data=fixdata, random_state=test_rng,
        )

    if fixdata:
        training_data  = training_data.torch_dataset()
        test_data = test_data.torch_dataset()
        val_size = validation_data.torch_dataset()

    return training_data, validation_data, test_data
