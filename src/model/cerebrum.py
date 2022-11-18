#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'cerebral' network (rnn_with_readout) + helper classes
"""
import torch
import torch.nn as nn
import inspect

class module_extended(nn.Module):
    r"""helper Pytorch Module with added parent -> children methods
    """

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        for module in self.children():
            if isinstance(module, module_extended):
                module.set_batch_size(batch_size)

    def set_seqlen(self, seqlen):
        self.seqlen = seqlen
        for module in self.children():
            if isinstance(module, module_extended):
                module.set_seqlen(seqlen)

    def set_timestep(self, timestep):
        self.timestep = timestep
        for module in self.children():
            if isinstance(module, module_extended):
                module.set_timestep(timestep)

    def set_classification(self, classification):
        self.classification = classification
        for module in self.children():
            if isinstance(module, module_extended):
                module.set_classification(classification)
            
    def set_predict_last(self, predict_last):
        self.predict_last = predict_last
        for module in self.children():
            if isinstance(module, module_extended):
                module.set_predict_last(predict_last)


class readout_net(module_extended):
    r"""The readout as a flexible feedforward net as opposed to one linear layer
    Args:
        parameters of readout neural network (often taken from cerebellar parameters)

    """
    def __init__(self, input_size, hidden_size, output_size, 
                 num_hidden_layers=1, nfibres=None, 
                 zero_output=False, fixed=False, bias=False,
                 nonlin='relu'):
        super(readout_net, self).__init__()
        self.input_size = input_size    
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        
        self.hidden_size = hidden_size if num_hidden_layers > 0 else 0
        self.bias = bias
        
        if nonlin == 'relu':
            self.f = torch.relu #non-linearity of network
        elif nonlin == 'spike':
            thresh = 0 
            def spike(x):
                return (x > thresh).float()
            self.f = spike
        elif nonlin == 'none':
            def identity(x):
                return x
            self.f = identity
        elif nonlin == 'sigmoid':   
            self.f = torch.sigmoid
        else:
            raise ValueError("unrecognised cerebellar non-linearity {}".format(nonlin))
            
        self.dropout = nn.Dropout(p=0.2)
        
        self.init_weights()

    def init_weights(self):
        top_layer_dim = self.output_size if self.num_hidden_layers == 0 else self.hidden_size
        
        self.input_trigger = torch.nn.Linear(
            in_features=self.input_size, out_features=top_layer_dim, bias=self.bias
        )

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=(
                    self.hidden_size if layer_index < self.num_hidden_layers - 1 else self.output_size
                ),
                bias=self.bias
            )
            for layer_index in range(self.num_hidden_layers)
        ])
    
    def forward(self, input):        
        pred = self.input_trigger(input)
        
        pred = self.dropout(pred)

        for layer in self.layers:         
            pred = layer(self.f(pred))
        
        return pred    

class rnn_with_readout(module_extended):
    r"""cerebral network; RNN + readout
    """
    def __init__(self, rnn, output_size, readout_params=None, early_target=None,
                 nreadout=1):
        """ 
        Args:
            rnn: predefined recurrent neural network (e.g. LSTM)
            output_size: readout size
            readout_params: parameters which define readout network. If none then
            just a linear layer
        """
    
        super(rnn_with_readout, self).__init__()
        self.rnn = rnn      
        self.init_readout(rnn.hidden_size, output_size, readout_params, nreadout=nreadout)
        self.early_target = early_target

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.readout.reset_parameters()

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def output_size(self):
        return self.linear.weight.shape[0]

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def n_layers(self):
        return self.rnn.num_layers
    
    def init_readout(self, hidden_size, output_size, readout_params=None,
                     nreadout=1):
        if readout_params is None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:                                  
            self.readout = readout_net(**readout_params, input_size=hidden_size,
                                              output_size=output_size) 
        
    def set_readout(self, readout_number=0):
        self.readout = self.all_readouts[readout_number]

    def forward(self, input, hidden=None, return_whole=False):
        output, hidden = self.rnn(input, hidden)
        
        if input.shape[1] == 1:
            output = output.squeeze(1)
        
        if output.ndim == 2:
            pred = self.readout(output)
        elif self.training and self.early_target is not None:
            tsteps = input.size(1)
            pred = [self.readout(output[:, i, :]) for i in range(self.early_target, tsteps)]
            pred = torch.stack(pred, dim=1)
        elif self.predict_last and not return_whole:
            pred = self.readout(output[:, -1, :])
        else:
            tsteps = input.size(1)
            pred = [self.readout(output[:, i, :]) for i in range(tsteps)]
            pred = torch.stack(pred, dim=1)
            
        return pred, hidden


readout_param_names = list(inspect.getfullargspec(readout_net))[0]

        