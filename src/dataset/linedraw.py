#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
line-drawing task (see Figs 2,3 in paper)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_circle_points(npoints, mag=10, shift_shuffle=False, include_zero=False):
    nedge = (npoints -1) if include_zero else npoints
    
    angles = np.linspace(0, 2*np.pi, nedge, endpoint=False)
    
    if shift_shuffle:
        #move each point by a half
        offset = (angles[1] - angles[0])/2
        angles = angles + offset
    
    coords = np.vstack((np.sin(angles), np.cos(angles))).T
    coords = mag * coords
    
    if include_zero:
        coords = np.concatenate((np.zeros((1, 2)), coords), axis=0)

    return coords

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

def draw_2dline(seqlen, end_point, start_point=None):
    #pick a start point and end point, 
    #and make sequence of points between them
    if start_point is None: 
        start_point = np.zeros(2) #start at origin
    
    seqx = np.linspace(start_point[0], end_point[0], num=seqlen)
    seqy = np.linspace(start_point[1], end_point[1], num=seqlen)
    
    line = np.vstack((seqx, seqy)).T  
    return line

def draw_semi_ellipse(seqlen, end_point, mag, arc=0, start_point=None):
    if start_point is None: 
        start_point = np.zeros(2) #start at origin
    
    if np.array_equal(start_point, end_point):
        seqx = np.linspace(start_point[0], end_point[0], num=seqlen)
        seqy = np.linspace(start_point[1], end_point[1], num=seqlen)
        line = np.vstack((seqx, seqy)).T  
        return line
    
    centre = 0.5 * (start_point + end_point)
      
    adj = end_point[0] - start_point[0]
    opp = end_point[1] - start_point[1] 
    
    alpha = np.arctan(opp/adj) #sohcahtoa!
    
    a = mag/2
    b = a * arc
    
    startx = centre[0] + a * np.cos(alpha)
    starty = centre[1] + a * np.sin(alpha)
    if np.isclose(startx, 0) and np.isclose(starty, 0):
        t = np.linspace(0, np.pi, seqlen, endpoint=True)
    else:
        t = np.linspace(np.pi, 2 * np.pi, seqlen, endpoint=True)
    
    seqx = centre[0] + a * np.cos(t) * np.cos(alpha) - b*np.sin(t) * np.sin(alpha)
    seqy = centre[1] + a * np.cos(t) * np.sin(alpha) + b*np.sin(t) * np.cos(alpha)
    
    line = np.vstack((seqx, seqy)).T  
    line[0] = 0. 
    
    return line

def define_targets(npoints, seqlen, mag=10, arc=0, equal_spacing=False,
                   shift_shuffle=False):
    
    targets = np.zeros((npoints, seqlen, 2)) 
    circle_points = get_circle_points(npoints, mag=mag, shift_shuffle=shift_shuffle, include_zero=True)
    
    for i in range(npoints):
        if equal_spacing:
            targets[i] = draw_2dline(seqlen, end_point=circle_points[i])
        else:
            curve = draw_semi_ellipse(seqlen, end_point=circle_points[i],
                                       mag=mag, arc=arc)        
            targets[i] = curve
    
    return targets
    

def get_input_target_pairs(input_D, seqlen, min_value, max_value, npoints, mag=10,
                           arc=0, equal_spacing=False, shift_shuffle=False):
    inputs = define_inputs(input_D, min_value, max_value, npoints)
    targets = define_targets(npoints, seqlen, mag=mag, arc=arc, 
                             equal_spacing=equal_spacing,
                             shift_shuffle=shift_shuffle)
    
    if shift_shuffle:
        #shuffle around
        shuffle_term = int(inputs.shape[0]/2)
        inputs = np.roll(inputs, shuffle_term, 0)
        targets = np.roll(targets, shuffle_term, 0)
    
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

def load_linedraw(
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
    npoints=2, 
    mag=1,
    arc=0,
    equal_spacing=False,
    shift_shuffle=False
):
    
    if input_D > 1:
        assert min_value == 0, 'not ready for non-zero min value'
        assert max_value == 1, 'not ready for non-one max value'
        
    assert arc <= 1, 'arc must be less than 1'
    
    if equal_spacing:
        assert arc == 0, 'for equal spacing require straight line (arc=0)'
        
    input_vals, target_vals = get_input_target_pairs(input_D, seq_len, min_value, max_value, npoints,
                                                     mag=mag, arc=arc, equal_spacing=equal_spacing,
                                                     shift_shuffle=shift_shuffle)
    
    N = int(training_size * (1 - train_val_split))
    val_size = training_size - N

    if random_state is None:
        train_rng = np.random.randint(2**16-1)
        val_rng = np.random.randint(2**16-1)
        test_rng = np.random.randint(2**16-1)
    else:
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
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
