#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .cerebro_cerebellum import cerebro_cerebellum
except:
    from cerebro_cerebellum import cerebro_cerebellum
    
    
class cc_sleep(cerebro_cerebellum):
    r"""cortico-cerebellar network during consolidation ('sleep')
        see Fig 8 in paper
    """
    def __init__(self, config_fn, input_size, output_size, 
                 cereb_neo_learning_ratio=1, decay_rate=0.1,
                 h_power=1, least_squares=False):  
        """
        Arguments
        ----------
        config_fn : initial model config (for model already trained on some task)
        input_size : model input size
        output_size : model output size
        cereb_neo_learning_ratio : ratio of cerebellar to cortical consolidation 
                                    learning rate
        decay_rate : rate of decay for cerebellar-to-cortical weights 
                    (also indirectly defines cortical learning rate)
        h_power : used to define bio learning rule (set as 1 in paper)
        least_squares : use optimal least square solution
        """
        
        with open(config_fn) as json_file:
            config = json.load(json_file)         
        model_config = config['model']
        model_config['input_size'] = input_size
        model_config['output_size'] = output_size
        
        super(cc_sleep, self).__init__(**model_config)
        
        self._sleep_mode(True)
        self.cerebellum.eval()
        self.sleeping_errors = []
                
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.cereb_neo_learning_ratio = cereb_neo_learning_ratio
        
        self.decay_rate = decay_rate
        self.h_power = h_power
        
        self.least_squares = least_squares
        
        if self.least_squares:
            self.all_hiddens = torch.zeros((13, 10, self.hidden_size))
            self.all_cereb_inputs = torch.zeros((13, 10, self.hidden_size))
                
        self.rnn_grad_acc = torch.zeros_like(self.cerebrum.rnn.w_hh)
                
    def store_orig_weights(self):
        self.hh_orig = self.cerebrum.rnn.w_hh.clone()
        self.cw_orig = self.cerebrum.rnn.w_ih[:, self.input_size_raw:].clone()
        
        
    def compute_sleeping_error(self, h, c):
        rec_input_orig = torch.mm(h, self.hh_orig.t())
        cereb_input_orig = torch.mm(c, self.cw_orig.t())
        combined_orig = rec_input_orig + cereb_input_orig
        
        rec_new = torch.mm(h, self.cerebrum.rnn.w_hh.t()) 
        
        cossim_all = cosine_similarity(combined_orig, rec_new)
        cossim = np.mean(np.diag(cossim_all))
        
        return cossim
        
    def _sleep_mode(self, sleep):
        self.sleep_mode = sleep
        if sleep:
            self.run = self.run_sleep
        else:
            self.run = self.run_with_cerebellar_feedback
    
    def sleep_learn(self, h, c):
        if torch.sum(c) == 0:
            return
        
        error = self.compute_sleeping_error(h, c)
        self.sleeping_errors.append(error)
        
        cereb_input = torch.mm(c, self.cerebrum.rnn.w_ih[:, self.input_size_raw:].t())
        
        if self.least_squares:
            self.all_hiddens[self.timestep-2] = h
            self.all_cereb_inputs[self.timestep-2] = cereb_input
            return
                
        power_helper = torch.abs(h)**self.h_power
        
        sum_h_power = torch.sum(power_helper, dim=1)
        
        div = torch.div(cereb_input, sum_h_power[:, None])

        power_helper = torch.mul(torch.sign(h), torch.abs(h)**(self.h_power-1))
        magic = torch.bmm(power_helper[:, :, None], div[:, None, :])
        rnn_grad = torch.mean(magic, dim=0)
        rnn_grad = rnn_grad.t()
        
        self.rnn_grad_acc += rnn_grad
    
    def run_sleep(self, input, hidden=None, latter_trunc=False, **kwargs):
        mc = input.new_zeros(size=(self.batch_size, self.output_size))
        pfc = input.new_zeros(size=(self.batch_size, self.hidden_size))
        targ = None
        cereb_out = None
        
        for t in range(self.seqlen):   
            self.set_timestep(t)
            inp_raw = input[:,t]
            
            if t > 0 or latter_trunc:
                if t > 1:
                    self.sleep_learn(pfc, cereb_out)  
                pfc = hidden[0][-1] if 'LSTM' in self.rnn_type or 'eprop' in self.rnn_type else hidden[-1]  
            
            cer_info = {'PFC': pfc, 'MC': mc, 
                            'targ': targ, 'inp': inp_raw}
            
            cereb_out = self.get_cereb_output(cer_info)
            inp = torch.cat((inp_raw, cereb_out), dim = 1).unsqueeze(1)
            mc, hidden = self.cerebrum(inp, hidden=hidden) 
            

        if self.least_squares:
            all_hiddens = self.all_hiddens.reshape(-1, self.hidden_size)
            all_cereb_inputs = self.all_cereb_inputs.reshape(-1, self.hidden_size)            
            self.rnn_grad_acc =  torch.lstsq(all_cereb_inputs, all_hiddens).solution[:self.hidden_size].t()
                  
        self.cerebrum.rnn.w_hh += self.decay_rate*self.rnn_grad_acc * (1/self.cereb_neo_learning_ratio) #recurrent weights
        self.cerebrum.rnn.w_ih[:, self.input_size_raw:] *= 1-self.decay_rate #cerebello-cortical weights

        torch.Tensor.zero_(self.rnn_grad_acc)

        return mc, hidden, None