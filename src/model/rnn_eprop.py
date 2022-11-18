#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leaky rnn with eprop: see Bellec et al. 2019
"""
import math
import torch

try:
    from .eprop import eprop
except:
    from eprop import eprop

class rnn_eprop(eprop):
    def __init__(self, alpha=0.1, *args, **kwargs):
        super(rnn_eprop, self).__init__(*args, **kwargs)
        self.alpha = alpha
        
        self.f = torch.tanh
        self.f_prime = lambda x: 1 - torch.tanh(x)**2
        self.cort_ablate = False #cortical ablation (cf. Fig S15 in paper)
        
    @property
    def gate_size(self):
        return self.hidden_size       
            
    def init_buffers(self):
        r"""Using buffers to contain states + eligibility trace values. 
        """
        
        shape = (self.batch_size, self.hidden_size)
                
        self.register_buffer('cell_state', torch.zeros(shape), persistent=False)
        self.register_buffer('out_state', torch.zeros(shape), persistent=False)
        
        for weight_type, s in zip(self.plastic_weights, [(self.input_size,), 
                                                 (self.hidden_size,), ()]):
            
            name = weight_type          
            name += '_evec'
            name_trail = name + '_trail'
            
            self.register_buffer(name, torch.zeros(shape + s), persistent=False)
            if self.trails_required:
                if self.predict_last:
                    if self.backprop:
                        trail_length = self.seqlen
                    elif self.early_target is not None:
                        trail_length = self.seqlen - self.early_target + 1
                    else:
                        trail_length = 1
                else:
                    trail_length = self.seqlen
                trail_shape = (trail_length,) + shape + s

                self.register_buffer(name_trail, torch.zeros(trail_shape),
                                     persistent=False)
                
        if self.trails_required:
            trail_shape = (trail_length,) + shape
            self.register_buffer('out_cell_grad_trail', torch.zeros(trail_shape),
                                 persistent=False)

    def store_evecs(self):
        r"""If waiting for backpropagated learning signals then trail of eligibility traces 
            will be stored and correctly matched to them
        """
        hook_counter = self.seqlen - self.timestep - 1

        for weight_type in self.plastic_weights:
            name = weight_type
            name += '_evec'
            evec_now = getattr(self, name)
            

            evec_trail = getattr(self, name + '_trail')           
            evec_trail[hook_counter] = evec_now
            
        out_cell_grad = self.f_prime(self.cell_state)  
        self.out_cell_grad_trail[hook_counter] = out_cell_grad
                
            
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
        self.compute_states()
    
    def compute_states(self):
        self.cell_state_old = self.cell_state
        self.out_state_old = self.out_state        

        inp = torch.mm(self.input_t, self.w_ih.t()) \
                    + torch.mm(self.out_state, self.w_hh.t()) 
                    
        self.cell_state = self.alpha * self.cell_state + inp
                            
        self.out_state = self.f(self.cell_state)  
        
        
        if self.cort_ablate:
            self.out_state[:, self.cort_abl_mask] = 0
                
                
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
        self.hook_counter += 1               
        out_cell_grad_gen = self.out_cell_grad_trail[self.hook_counter]
        for i, weight in enumerate(self.parameters()):
            if self.weight_types[i] not in self.plastic_weights:
                continue
            
            if i < 2:
                out_cell_grad = out_cell_grad_gen[:, :, None]
                ls = outstate_grad[:, :, None]               
            else:
                out_cell_grad = out_cell_grad_gen
                ls = outstate_grad
            
            evec_trail = getattr(self, self.weight_types[i] + '_evec_trail')            
            etrace = evec_trail[self.hook_counter] * out_cell_grad
            
            grad = torch.sum(etrace * ls, dim=0)
            
            if self.weight_types[i] in self.plastic_weights:              
                weight.grad += grad   
           
        
    def update_traces(self):
        r"""update the eligibility traces
        """    
        post_term = self.input_t[:, None, :].repeat((1, self.hidden_size, 1))
        self.i_evec = self.i_evec * self.alpha + post_term
                            
        if len(self.plastic_weights) > 1:      
            post_term = self.out_state_old[:, None, :].repeat((1, self.hidden_size, 1))                     
            self.r_evec = self.r_evec * self.alpha + post_term
                
            if self.bias:
                self.b_evec = self.b_evec * self.alpha + torch.ones_like(self.b_evec)        


    def enfore_evec_decay(self):
        r"""Enforce a decay on the eligibility vectors.  
        """
        for weight_type in self.plastic_weights:
            evec = getattr(self, weight_type + '_evec')
            evec *= self.evec_scale 
            
    def prepare_cort_ablation(self, fraction_ablation=0.5):
        nablate = int(fraction_ablation * self.hidden_size)
        self.cort_abl_mask = torch.arange(nablate)

    def ablate_cort_inp(self, ablate):
        self.cort_ablate = ablate