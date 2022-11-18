#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obtaining error gradients with eligibility traces (eprop; see Bellec et al. 2019)
"""

import math
import torch
from torch.nn.parameter import Parameter

try:
    from .cerebrum import module_extended
except:
    from cerebrum import module_extended


class eprop(module_extended):
    r"""An RNN network which trains using eligibility traces. See Bellec 
        et al. 2019 for details.        
    """
    def __init__(self, input_size, hidden_size, store_traces=False, 
                 evec_scale=None, fixed_rnn=False, bias=True, 
                 early_target=None, backprop=False):
        super(eprop, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.store_traces = store_traces
        self.evec_scale = evec_scale
        self.bias = bias

        self.record_etrace = False
        
        self.early_target = None
        
        self.fixed_rnn = fixed_rnn
        self.backprop = backprop
            
        self.init_weights()
        
        if fixed_rnn == False:
            self.plastic_weights = self.weight_types.copy()
        elif fixed_rnn == 'inp_only':
            self.plastic_weights = ['i']
        else:
            self.plastic_weights = []
        
        self.trails_required = True if self.plastic_weights else False
        
        self.hook_counter = -1
        
    @property
    def gate_size(self):
        raise NotImplementedError        

    def init_weights(self):
        gate_size = self.gate_size   
        w_ih = Parameter(torch.Tensor(gate_size, self.input_size))
        w_hh = Parameter(torch.Tensor(gate_size, self.hidden_size))

        params = [w_ih, w_hh]
        param_names = ['w_ih', 'w_hh']
        
        self.weight_types = ['i', 'r']
        
        if self.bias:
            b = Parameter(torch.Tensor(gate_size))   
            params.append(b)
            param_names.append('b')
            self.weight_types.append('b')
            

        for name, param in zip(param_names, params):
            setattr(self, name, param)
            
        
        self.requires_grad_(requires_grad=False)
        self.reset_parameters()    
    
    def init_buffers(self):
        r"""Using buffers to contain LSTM states + eligibility trace values. 
        """
        raise NotImplementedError
 
    def reset_buffers(self, new_batch_size=True):
        if new_batch_size:
            new_shape = (self.batch_size,) + self.cell_state.shape[1:]
            self.cell_state = self.cell_state.new_zeros(size=new_shape)   
            self.out_state = self.out_state.new_zeros(size=new_shape)           
            new_batch_size = False
            
        if self.training:
            for name, buffer in self.named_buffers():
                setattr(self, name, torch.zeros_like(buffer))

        elif not new_batch_size:
            self.cell_state = torch.zeros_like(self.cell_state)
            self.out_state = torch.zeros_like(self.out_state)
        
        self.hook_counter = -1


    def store_evecs(self):
        r"""If waiting for backpropagated learning signals then trail of eligibility traces 
            will be stored and correctly matched to them
        """
        raise NotImplementedError
        
    def forward(self, input, hidden=None, target=None, latter_trunc=False, 
                noise_thrown=False, record_etrace=False):
        r"""Like LSTM forward, but will include learning aspect as well
        """             
        if hidden is not None:
            self.set_hidden(hidden)
        
        input = input.squeeze(1)
        
        #check if there's just one timestep's worth 
        if input.ndim == 2:
            self.input_t = input
            self.forward_step()  

            if self.training and self.fixed_rnn != True:     
                self.accumalate_gradients()   

            output = self.out_state
        else:    
            output = self.run_over_seq(input)
            
        hidden = self.get_hidden()

        return output, hidden

    
    def forward_step(self):
        raise NotImplementedError 
    
    def set_hidden(self, hidden, noise_thrown=False, latter_trunc=False): 
        if self.backprop:
            self.out_state, self.cell_state = (hidden[0].squeeze(0), hidden[1].squeeze(0))
        else:
            self.out_state, self.cell_state = (hidden[0].squeeze(0).detach(), hidden[1].squeeze(0).detach())
    
    def get_hidden(self):
        return self.out_state.unsqueeze(0), self.cell_state.unsqueeze(0)
    
    def run_over_seq(self, input):  
        r"""run over multiple timesteps (usually whole sequence)
        """       
        output = input.new_zeros(size=(self.batch_size, self.seqlen, self.hidden_size))
        for t in range(self.seqlen):
            self.timestep = t

            if self.store_traces:
               self.inpr_evec_trail[t] = self.inpr_evec
               
            self.input_t = input[:,self.timestep]
            self.forward_step()
            
            if self.training and self.fixed_rnn != True:     
                self.accumalate_gradients()             
            output[:, t, :] = self.out_state
            
            if not self.backprop:
                self.out_state = self.out_state.detach()
            
        return output
        
                
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters(): 
            torch.nn.init.uniform_(weight, -stdv, stdv) 
            weight.grad = torch.zeros_like(weight)
        
            
    def accumalate_gradients(self):
        r"""Update and store eligibility traces and prepare backpropagated learning signal
        """    
        self.update_traces()
        if self.ls_available():  
            self.store_evecs()   
            self.out_state.requires_grad_()
                            
            self.out_state.register_hook(self.backprop_hook)

       
    def backprop_hook(self, outstate_grad):
        r"""will be applied once backward() reaches the out_state - this is the
        'learning signal' that will be mixed with the respective eligibility trace 
        """          
        raise NotImplementedError 
        
    def ls_available(self):
        r"""is/will a learning signal be available for this current timestep?
        """         
        ls_available = False
        if not self.training:
            ls_available = False
        elif not self.predict_last:
            ls_available = True
        elif self.timestep == self.seqlen - 1:
            ls_available = True      
        elif self.early_target is not None and self.timestep >= self.early_target - 1:
            ls_available = True
        elif self.backprop:
            ls_available = True
        
        return ls_available
   
        
    def update_traces(self):
        r"""update the eligibility traces
        """    
        raise NotImplementedError       

    def enfore_evec_decay(self):
        r"""Enforce a decay on the eligibility vectors.  
        """
        raise NotImplementedError 
        
    def update_plastic_sites(self, fixed_rnn):
        self.fixed_rnn = fixed_rnn
        if fixed_rnn == False:
            self.plastic_weights = self.weight_types.copy()
        elif fixed_rnn == 'inp_only':
            self.plastic_weights = ['i']
        else:
            self.plastic_weights = []
            
        self.trails_required = True if self.plastic_weights else False
        
        self.init_buffers()
        

            

        
        