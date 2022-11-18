#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lstm with eprop: see Bellec et al. 2019
"""
import torch

try:
    from .eprop import eprop
except:
    from eprop import eprop

class LSTM_eprop(eprop):
    def __init__(self, alpha=0.1, *args, **kwargs):
        super(LSTM_eprop, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.init_weights()
        
    @property
    def gate_size(self):
        return 4 * self.hidden_size 

    def join_hidden(self, hidden):
        hidden = torch.cat(hidden, dim=1)
        return hidden

    def split_hidden(self, hidden):
        (h, c) = hidden.chunk(2, dim=1)
        hidden = (h.contiguous(), c.contiguous())
        return hidden               

            
    def init_buffers(self):
        r"""Using buffers to contain LSTM states + eligibility trace values. 
        """
        
        shape = (self.batch_size, self.hidden_size)
                
        self.register_buffer('cell_state', torch.zeros(shape), persistent=False)
        self.register_buffer('out_state', torch.zeros(shape), persistent=False)
        
        for gate in ['inp', 'forget', 'out', 'cand']:
            for weight_type, s in zip(self.plastic_weights, [(self.input_size,), 
                                                     (self.hidden_size,), ()]):
                
                name = gate + weight_type
                if gate == 'out':                       
                    name += '_etrace'
                else:
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
        
        for gate in ['inp', 'forget', 'out', 'cand']:
            for weight_type in self.plastic_weights:
                name = gate + weight_type
                if gate == 'out':                       
                    name += '_etrace'
                else:
                    name += '_evec'
                evec_now = getattr(self, name)
                evec_trail = getattr(self, name + '_trail')           
                evec_trail[hook_counter] = evec_now
        
        out_cell_grad = self.out_gate * (1 - torch.tanh(self.cell_state)**2)
        
        self.out_cell_grad_trail[hook_counter] = out_cell_grad
        
    def forward_step(self):
        self.compute_gates()
        self.compute_states()            
        
    def compute_gates(self):  
        gates = torch.mm(self.input_t, self.w_ih.t()) \
                    + torch.mm(self.out_state, self.w_hh.t()) 
                            
        if self.bias:
            gates += self.b
                    
        ingate, forgetgate, candgate, outgate = gates.chunk(4, 1)

        
        self.inp_gate = torch.sigmoid(ingate)
        self.forget_gate = torch.sigmoid(forgetgate)
        self.out_gate = torch.sigmoid(outgate)
        self.cand_gate = torch.tanh(candgate)         

        if self.alpha is not None:
            self.forget_gate = self.alpha * torch.ones_like(forgetgate)
    
    def compute_states(self):
        self.cell_state_old = self.cell_state
        self.out_state_old = self.out_state
        
        self.cell_state = (self.forget_gate * self.cell_state) + (self.inp_gate * self.cand_gate)              
        self.out_state = self.out_gate * torch.tanh(self.cell_state)                
        
                   
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
            
            for j, gate in enumerate(['inp', 'forget', 'cand', 'out']):                
                if gate == 'out':
                    etrace_trail = getattr(self, gate + self.weight_types[i] + '_etrace_trail')  
                    etrace = etrace_trail[self.hook_counter] 

                else:
                    evec_trail = getattr(self, gate + self.weight_types[i] + '_evec_trail')                    
                    etrace = evec_trail[self.hook_counter] * out_cell_grad
                
                grad = torch.sum(etrace * ls, dim=0)
                
                if self.weight_types[i] in self.plastic_weights:              
                    weight.grad[j*self.hidden_size: (j+1)*self.hidden_size] += grad   
           
        
    def update_traces(self):
        r"""update the eligibility traces
        """    
        if self.evec_scale is not None:
            self.enfore_evec_decay()
        self.update_inp_evecs()
        self.update_forget_evecs()
        self.update_output_trace()
        self.update_cand_evecs()           
                
    def update_inp_evecs(self):
        post_synapse_term = (self.cand_gate * self.inp_gate * (1 - self.inp_gate))
        self.inp_post_term = post_synapse_term

        self.inpi_evec = self.inpi_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.input_t[:, None, :])
                            
        if len(self.plastic_weights) > 1:                           
            self.inpr_evec = self.inpr_evec * self.forget_gate[:, :, None] \
                                + torch.bmm(post_synapse_term[:, :, None], 
                                            self.out_state_old[:, None, :])
    
            
            if self.bias:
                self.inpb_evec = self.inpb_evec * self.forget_gate \
                                    + post_synapse_term
                            
        
    def update_forget_evecs(self):
        post_synapse_term = (self.cell_state_old * self.forget_gate * (1 - self.forget_gate))
        self.forget_post_term = post_synapse_term

        self.forgeti_evec = self.forgeti_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.input_t[:, None, :])
                            
        if len(self.plastic_weights) > 1:                           
            self.forgetr_evec = self.forgetr_evec * self.forget_gate[:, :, None] \
                                + torch.bmm(post_synapse_term[:, :, None], 
                                            self.out_state_old[:, None, :])
                
            if self.bias:
                self.forgetb_evec = self.forgetb_evec * self.forget_gate \
                                + post_synapse_term
 
    
    def update_output_trace(self):
        post_synapse_term = (torch.tanh(self.cell_state) * self.out_gate  \
                             * (1 - self.out_gate))  
        self.output_post_term = post_synapse_term #save for learning signal later..

        self.outi_etrace = torch.bmm(post_synapse_term[:, :, None], 
                                      self.input_t[:, None, :])
        
        if len(self.plastic_weights) > 1: 
            self.outr_etrace = torch.bmm(post_synapse_term[:, :, None], 
                                          self.out_state_old[:, None, :])
            
            if self.bias:
                self.outb_etrace = post_synapse_term
        

    def update_cand_evecs(self):
        post_synapse_term = (self.inp_gate * (1 - self.cand_gate**2))
        self.cand_post_term = post_synapse_term
        
        self.candi_evec = self.candi_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.input_t[:, None, :])
                            
        if len(self.plastic_weights) > 1: 
            self.candr_evec = self.candr_evec * self.forget_gate[:, :, None] \
                                + torch.bmm(post_synapse_term[:, :, None], 
                                            self.out_state_old[:, None, :])
             
            if self.bias:                    
                self.candb_evec = self.candb_evec * self.forget_gate \
                                + post_synapse_term

    def enfore_evec_decay(self):
        r"""Enforce a decay on the eligibility vectors.  
        """
        for gate in ['inp', 'forget', 'cand']:
            for weight_type in self.plastic_weights:
                evec = getattr(self, gate + weight_type + '_evec')
                evec *= self.evec_scale 