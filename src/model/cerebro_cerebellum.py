#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cerebro-cerebellar network with cerebellum driving cortical dynamics; 
see https://www.biorxiv.org/content/10.1101/2022.11.14.516257v1
"""
import os, sys
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import copy
from functools import wraps

try:
    from .rnn_eprop import rnn_eprop
    from .lstm_eprop import LSTM_eprop
    from .cerebellum import cerebellum
    from .cerebrum import module_extended, rnn_with_readout, readout_param_names
except:
    from lstm_eprop import LSTM_eprop
    from rnn_eprop import rnn_eprop
    from cerebellum import cerebellum
    from cerebrum import module_extended, rnn_with_readout, readout_param_names


class cerebro_cerebellum(module_extended):
    r"""A cerebro-cerebellar neural network, where the cerebrum (or 'cortex') is an RNN
    and the cerebellum is a feedforward net: 
    
    If apply_cerebellum is False is a normal RNN with a readout;
    if True then we have at timestep t:
        1. Cerebellum is fed a mixture of variables seen at t-1 (e.g cerebral activity)
        and trained to associate with some later seen variables. The cerebellar input/target
          defined by cereb_params['pons_sizes'] and cereb_params['IO_sizes'] respectively
        2. Cerebellar output conjoined with 'raw' input (original stimulus) to form 
        cerebral input
        3. Cerebral (RNN + readout) dynamics run for one timestep
        4. Repeat from 1 for next timestep 
    
    If apply_cerebellum is 'pseudo', then apply 1-4 but replacing cerebellar output
    with (cortical) readout.
    
    """
    def __init__(self, rnn_type, hidden_size, input_size, n_layers, output_size,
             evec_scale=None, apply_cerebellum=False, cereb_params=None,
             readout_from_cereb=False, 
             fixed_rnn=False, bias=True, rnn_weight_scale=None, alpha=0.1,
             backprop=False):
        """
        Arguments
        ----------
        rnn_type : type of RNN to use; valid values are
            'LSTM': LSTM network (trained with backprop)
            'LSTM_eprop' LSTM network (trained with eligibility traces)
            'RNN_tanh': RNN network with tanh non-linearity (trained with backprop)
            'RNN_relu': RNN network with relu non-linearity (trained with backprop)
            'RNN_eprop': RNN network with tanh/relu non-linearity (trained with eligibility traces)
        hidden_size : RNN size
        input_size : (external) input size to RNN
        n_layers : number of RNN layers (only valid for non eprop models)
        output_size : RNN readout size
        evec_scale : if not None scale eligibility traces to value (unused in paper)
        apply_cerebellum: network architecture; valid values are
            False: no feedback to RNN
            True: cerebellar feedback to RNN
            'pseudo': readout feedback to RNN (e.g. FORCE)
        cereb_params : parameters for cerebellar network
        readout_from_cereb : use cerebellar network as readout
        fixed_rnn: plasticity in RNN; valid values are
            False: unconstrained plasticity in RNN
            True: no plasticity in RNN
            'inp_only': plasticity at RNN input weights only
        bias : bias weights in RNN
        rnn_scale : if not None scale RNN weights by value (unused in paper)
        alpha : cerebral internal memory for (leaky) cerebral dynamics: h_t = alpha*h_t + I_t
        backprop : use backprop (through time) in eprop models
        """
        
        super(cerebro_cerebellum, self).__init__()   
                        
        self.input_size_raw = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.bias = bias
        
        cereb_params = copy.deepcopy(cereb_params)
        self.cereb_output_size = self.get_cereb_output_size(apply_cerebellum, cereb_params)
        
        if 'temp_basis' in cereb_params.keys() and cereb_params['temp_basis']: 
            assert apply_cerebellum != 'pseudo', 'can only use temporal basis with cerebellar feedback'
            self.cereb_output_size *= (cereb_params['IO_delays']['targ_IO_delay'] + 1)
        
        readout_params = self.readout_from_cereb(cereb_params) if readout_from_cereb else None
        
        if apply_cerebellum == 'pseudo':
            apply_cerebellum = False 
            self.pseudo = True
        else:
            self.pseudo = False
            
        self.init_cerebrum(readout_params, rnn_weight_scale=rnn_weight_scale,
                           evec_scale=evec_scale, fixed_rnn=fixed_rnn, 
                           alpha=alpha, backprop=backprop)   
        
            
        self.apply_cerebellum = apply_cerebellum
        if self.apply_cerebellum:
            new_cereb_params = self.configure_pons_and_IO(cereb_params)
            self.init_cerebellum(new_cereb_params)
            
        #backprop cerebellum + readout, and rnn in general if LSTM
        self.requires_grad_(False)
        if 'eprop' in self.rnn_type or fixed_rnn:
            self.cerebrum.readout.requires_grad_(True)
            if fixed_rnn == 'inp_only' and 'eprop' not in self.rnn_type:
                self.cerebrum.rnn.weight_ih_l0.requires_grad_(True)
                if self.bias:
                    self.cerebrum.rnn.bias_ih_l0.requires_grad_(True)
                    self.cerebrum.rnn.bias_hh_l0.requires_grad_(True)
        else:
            self.cerebrum.requires_grad_(True)
        
        if apply_cerebellum:
            self.cerebellum.requires_grad_(True)
            if self.cerebellum.num_hidden_layers > 0 and new_cereb_params['fixed_MF']:
                self.cerebellum.input_trigger.requires_grad_(False)
                            
            self.run = self.run_with_cerebellar_feedback   
        elif self.pseudo:
            self.run = self.run_with_readout_feedback
        else:
            self.run = self.run_without_feedback   
                       
    def readout_from_cereb(self, cereb_params):
        r"""Build readout network based on cerebellar parameters
        (no feedback + cerebellar readout in paper)
        """
        readout_params = cereb_params.copy()
        for key in cereb_params.keys():
            if key not in readout_param_names:
                readout_params.pop(key)
        
        return readout_params
            
    def get_cereb_origin_sizes(self, origin_name):
        r"""Original sizes of different origins used for cerebellar input/targets
        PFC: RNN out state; MC: readout; inp: external input; targ: task target 
        """
        if origin_name == 'PFC':
            origin_size = self.hidden_size
        elif origin_name == 'inp':
            origin_size = self.input_size_raw
        elif origin_name in ['MC', 'targ']:
            origin_size = self.output_size                                 
        else:
            raise ValueError("invalid origin {}".format(origin_name))
        
        return origin_size
    
    
    def get_cereb_output_size(self, apply_cerebellum, cereb_params):
        r"""extra input size given to the cortex from the (perhaps pseudo)
        cerebellar output (the 'cerebellar nuclei')
        """
        cereb_output_size = 0
        if apply_cerebellum == True:
            for name, size in cereb_params['IO_sizes'].items():
                name = name.replace('_IO_size', '')
                if size is None:
                    size = self.get_cereb_origin_sizes(name)
                cereb_output_size += size
        elif apply_cerebellum == 'pseudo':
            cereb_output_size = self.output_size
            
        return cereb_output_size
    
    
    def init_cerebrum(self, readout_params=None, rnn_weight_scale=None,
                      evec_scale=None, fixed_rnn=False, alpha=0.1, 
                      backprop=False):
        r"""cerebrum = RNN (e.g. prefrontal cortex) + readout (e.g. motor cortex)
        """
        input_size = self.input_size_raw + self.cereb_output_size
        if self.rnn_type == 'LSTM':
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True,
                bias=self.bias,
            )     
        elif self.rnn_type == 'LSTM_eprop':
            rnn = LSTM_eprop(
                input_size=input_size, 
                hidden_size=self.hidden_size,
                evec_scale=evec_scale,
                fixed_rnn=fixed_rnn,
                bias=self.bias,
                alpha=alpha,
                backprop=backprop,
                )      
        elif self.rnn_type == 'RNN_eprop':
            rnn = rnn_eprop(
                input_size=input_size, 
                hidden_size=self.hidden_size,
                evec_scale=evec_scale,
                fixed_rnn=fixed_rnn,
                bias=self.bias,
                alpha=alpha,
                backprop=backprop
                )   
        elif self.rnn_type in ['RNN_tanh', 'RNN_relu']:
            nonlin = self.rnn_type[4:]
            rnn = nn.RNN(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True,
                bias=self.bias,
                nonlinearity=nonlin,
        )
            
        if rnn_weight_scale is not None:
            with torch.no_grad():
                for param in rnn.parameters():
                    param *= rnn_weight_scale        
        
        self.cerebrum = rnn_with_readout(rnn, self.output_size, readout_params=readout_params,
                                     )
    
    def configure_pons_and_IO(self, cereb_params=None, given_weights=False):
        r"""define the input/target to the cerebellum via the 'pons' and 'IO'
        if given pons_size/IO_size is 0, ignore origin
        if given pons_size/IO_size is None, make direct clone of origin (as used in paper)
        if given pons_size/IO_size > 0, apply fixed (possibly dimensionality
        reducing) weight from origin to pons
        
        args: 
            cereb_params: cerebellar parameters
            given_weights: if loading model from a state_dict, will need two 
            calls of this method - one to initialise correctly with cereb_params, 
            one to ensure fixed weights of old state_dict is used
        """
                 
        def get_trans(weight_name=None):            
            def compose2(f, g):
                return lambda x: f(g(x))
            
            detach_transform = lambda x: x.detach()     
            
            if weight_name is not None:
                weight = getattr(self, weight_name)
                weight_transform = lambda x: torch.mm(x, weight.t())
                main_func = compose2(detach_transform, weight_transform)          
            else:
                main_func = detach_transform
            
            func = lambda x: main_func(x) if x is not None else None
            
            return func
        
        if given_weights:

            for trans_type in ['pons_trans', 'IO_trans']:
                trans_dict = getattr(self, trans_type)
                for key in trans_dict.keys():
                    weight_key = key.replace('trans', 'weight')
                    if hasattr(self, weight_key):
                        print("Resetting {} according to {}".format(key, weight_key))
                        trans_dict[key] = get_trans(weight_key)
            
            return
        
        
        assert cereb_params is not None, "Applying cerebellum, set cerebellar parameters!" 
            
        new_sizes = []
        possible_origins = ['PFC', 'MC', 'targ', 'inp'] #RNN, readout, target, external input    
        for dict_name in ['pons_sizes', 'IO_sizes']:
            
            sizes = {}         
            size_dict  = cereb_params[dict_name]
            suffix = '_' + dict_name[:-1]
            
            trans_dict_name = dict_name.replace('sizes', 'trans')
            trans_dict = getattr(self, trans_dict_name) if given_weights else {}          
            for i, (key, value) in enumerate(size_dict.items()):
                assert suffix in key, 'invalid key name in pons_sizes {}'.format(key)
                origin_name = key.replace(suffix, '')
                assert origin_name in possible_origins, 'invalid origin {}'.format(origin_name)
                origin_size = self.get_cereb_origin_sizes(origin_name)                
                trans_name = origin_name + suffix.replace('size', 'trans')
                
                 
                weight_name = None
                if value is None:                   
                    sizes[key] = origin_size
                elif value == 0:
                    weight = -1
                elif value > 0:
                    weight_name = trans_name.replace('trans', 'weight') 
                    weight = Parameter(torch.Tensor(value, origin_size), requires_grad=False)
                    torch.nn.init.kaiming_uniform_(weight, a=np.sqrt(5))
                    sizes[key] = value
                    
                    self.register_buffer(weight_name, weight, persistent=True)
                    
                else:
                    raise ValueError("Invalid size {}".format(value))
                
                trans = get_trans(weight_name) 
                if value != 0:
                    trans_dict[trans_name] = trans   
                  
            if not given_weights:
                setattr(self, trans_dict_name, trans_dict)
            
            
            new_sizes.append(sizes)
        
        new_cereb_params = cereb_params.copy()
        new_cereb_params['pons_sizes'] = new_sizes[0]
        new_cereb_params['IO_sizes'] = new_sizes[1]
        
        if 'fixed_MF' not in new_cereb_params.keys():
            new_cereb_params['fixed_MF'] = False
        
        return new_cereb_params
        
    def init_cerebellum(self, cereb_params):       
        self.cerebellum = cerebellum(**cereb_params)            
                      
    def init_dataset_details(self, batch_size, seqlen, classification,
                             predict_last, timestep=-1, final_window=None):
        r"""dataset dependent model variables
        """
        self.set_batch_size(batch_size)
        self.set_seqlen(seqlen)
        self.set_classification(classification)
        self.set_predict_last(predict_last)
        self.set_timestep(timestep)      
        
        self.final_window = final_window
        
        if self.final_window:
            self.train = self.wrap_predlast(self.train)
            self.set_predict_last(False) #start with training..
        
        if 'eprop' in self.rnn_type:
            self.cerebrum.rnn.init_buffers()
        if self.apply_cerebellum:
            self.cerebellum.init_loss_fns()
          
    def init_batch(self, input):        
        r"""for each batch may need to reset buffers (e.g. eligibility traces) + batch_sizes
        """
        self.set_timestep(-1)            
        batch_size, seqlen = input.shape[:2]
        
        batch_size_change = False
        if batch_size != self.batch_size:
            batch_size_change = True
            self.set_batch_size(batch_size)
        if seqlen != self.seqlen:
            self.set_seqlen(seqlen)
        
        if 'eprop' in self.rnn_type:
            self.cerebrum.rnn.reset_buffers(batch_size_change)
        if self.apply_cerebellum:
            self.cerebellum.reset_buffers()
            
    def run_with_cerebellar_feedback(self, input, hidden=None, target=None, latter_trunc=False,
                            return_whole=False, ablate_cer_timestep=None): 
        """
        run cerebro-cerebellar dynamics
        """
        
        if not self.predict_last or return_whole:
            readout = input.new_zeros((self.batch_size, self.seqlen, self.output_size))
        
        readout_t = None           
        
        for t in range(self.seqlen):        
            if ablate_cer_timestep is not None and t == ablate_cer_timestep:
                self.cerebellum.force_zero_output(True)
            
            inp_raw = input[:,t]
            
            if self.training:
                if self.final_window:
                    targ = None if t <= self.seqlen - self.final_window else target[:, t + self.final_window - 1 - self.seqlen]
                    
                else:
                    targ = None if self.predict_last else target[:, t-1]
            else:
                targ = None
            
            self.set_timestep(t)
            

            #remember, the cerebellum works with the previous timesteps neocortical activity + current input
            if t > 0 or latter_trunc:                
                hidden_state = hidden[0][-1] if 'LSTM' in self.rnn_type or 'eprop' in self.rnn_type else hidden[-1]
                
                cer_info = {'PFC': hidden_state, 'MC': readout_t, 
                            'targ': targ, 'inp': inp_raw}
                cereb_out = self.get_cereb_output(cer_info)
            else:         
                pfc = input.new_zeros(size=(self.batch_size, self.hidden_size))
                mc = input.new_zeros(size=(self.batch_size, self.output_size))
                targ = None
                cer_info = {'PFC': pfc, 'MC': mc, 
                            'targ': targ, 'inp': inp_raw}
                
                cereb_out = self.get_cereb_output(cer_info)
                
            
            inp = torch.cat((inp_raw, cereb_out), dim = 1).unsqueeze(1)
            readout_t, hidden = self.cerebrum(inp, hidden=hidden)   
                
            if not self.predict_last or return_whole:
                readout[:, t] = readout_t


        targ = target if self.predict_last or target is None else target[:, -1]
        
                    
        hidden_state = hidden[0][-1] if 'LSTM' in self.rnn_type or 'eprop' in self.rnn_type else hidden[-1]   
        
        cer_info = {'PFC': hidden_state, 'MC': readout_t, 
                    'targ': targ, 'inp': input[:,-1]}   
        

        if self.seqlen > 1 or (self.cerebellum.save_outputs and self.cerebellum.n == self.cerebellum.recorded_outputs.shape[0]-1):
            self.get_cereb_output(cer_info, final_learning=True)
             
        
        if self.predict_last and not return_whole:
            readout = readout_t

       
        if ablate_cer_timestep is not None:
            self.cerebellum.force_zero_output(False) #undo ablation at end of task
        
        return readout, hidden, target
        

    def run_with_readout_feedback(self, input, hidden=None, target=None, latter_trunc=False,
                                   return_whole=False, ablate_cer_timestep=None):
        r"""use the readout for RNN feedback as opposed to cerebellar output
        """
        if not self.predict_last or return_whole:
            readout = input.new_zeros((self.batch_size, self.seqlen, self.output_size))
        
        readout_t = None
        for t in range(self.seqlen):
            inp_raw = input[:,t]            
            self.set_timestep(t)
            
            if t > 0 or latter_trunc:               
                readout_feedback = self.readout_t_cached.detach() if latter_trunc else readout_t.detach()
            else:
                readout_feedback = input.new_zeros(size=(self.batch_size, self.output_size))
                
            if self.classification:
                readout_feedback = torch.softmax(readout_feedback, dim=-1)
                
            inp = torch.cat((inp_raw, readout_feedback), dim = 1).unsqueeze(1)
            readout_t, hidden = self.cerebrum(inp, hidden=hidden)
            
            if t == 0 or latter_trunc:
                self.readout_t_cached = readout_t

            if not self.predict_last or return_whole:
                readout[:, t] = readout_t
        
        if self.predict_last and not return_whole:
            readout = readout_t
        
        return readout, hidden, target
    
    def run_without_feedback(self, input, hidden=None, target=None, latter_trunc=False,
                               return_whole=False, ablate_cer_timestep=None):
        """no feedback RNN"""
        
        readout, hidden = self.cerebrum(input, hidden=hidden, return_whole=return_whole)           
        return readout, hidden, target
    
    def forward(self, input, hidden=None, target=None, latter_trunc=False, return_whole=False,
                ablate_cer_timestep=None):
        """ 
        network forward pass
        args: 
            input: (external) input to model
            hidden: initial hidden state 
            target: task target (used to train cerebellum)
            latter_trunc: used in case want to reinitialise hidden state
            return_whole: return whole sequence of model readout
            ablate_cer_timestep: ablate cerebellar output at that timestep
        """        
        
        if not latter_trunc:
            self.init_batch(input)
        readout, hidden, target = self.run(input, hidden=hidden, target=target, latter_trunc=latter_trunc,
                                           return_whole=return_whole, ablate_cer_timestep=ablate_cer_timestep)
        

        if self.training:
            if self.final_window:
                readout = readout[:, -self.final_window:].permute(0, 2, 1)                        
            return readout, hidden, target
        else:
            return readout, hidden
            
    
    def get_cereb_output(self, cer_info, final_learning=False):        
        r"""get cerebellar output (and teach it)
        """
        pons = self.get_pons(cer_info)
        IO = self.get_IO(cer_info)
        cereb_output = self.cerebellum(pons, IO, final_learning=final_learning)
                
        return cereb_output
    
    def get_pons(self, cer_info):
        r"""activity at the pons - the input to the cerebellum
        """
        pons = []
        for key, act in self.pons_trans.items():
            origin = key.replace('_pons_trans', '')
            pons.append(act(cer_info[origin]))                  
        pons = torch.cat(pons, dim=1)
        
        return pons
    
    def get_IO(self, cer_info):
        r"""activity at the inferior olive - the target to the cerebellum
        """
        if not self.training:
            return None        
        else:
            IO = {}
            for key, act in self.IO_trans.items():
                origin = key.replace('_IO_trans', '')
                IO[origin] = act(cer_info[origin])

        return IO
    
    def wrap_predlast(self, func):
        r"""helper function to train delayed association task on final few timesteps,
        but only test at the last"""
        @wraps(func)
        def wrapper(training=True):
            self.set_predict_last(not training)
            return func(training)
        return wrapper
    
    def set_2task_model(self,):
        r"""use distinct cerebellar weights per task; see Fig 3 of paper"""      
        if self.apply_cerebellum:
            self.all_cerebellums = torch.nn.ModuleList([self.cerebellum, copy.deepcopy(self.cerebellum)])
        
    def share_pf1_to_pf2(self, number_to_switch, pf_overlap=None):
        r""" share ratio pf_overlap of cerebellar PF weights for each task"""
        if pf_overlap is None:
            return 
        orig_shape = self.cerebellum.layers[0].weight.shape
        
        pf_size = orig_shape[0] * orig_shape[1]
        overlap_size = int(pf_size * pf_overlap)
        self.pf_overlap_inds = torch.arange(overlap_size)
        
        with torch.no_grad():
            pf1_flat = self.cerebellum.layers[1-number_to_switch].weight.reshape(-1)
            pf2_flat = self.all_cerebellums[number_to_switch].layers[0].weight.data.reshape(-1)
            
            pf2_flat[self.pf_overlap_inds] = pf1_flat[self.pf_overlap_inds]
            
            self.all_cerebellums[number_to_switch].layers[0].weight.data = pf2_flat.reshape(orig_shape)
    
    def set_task_number(self, task_number, pf_overlap=None):
        r"""use cerebellar network specific to task/context"""
        if self.apply_cerebellum:
            self.share_pf1_to_pf2(task_number, pf_overlap=pf_overlap)
            self.cerebellum = self.all_cerebellums[task_number]
                
            
                