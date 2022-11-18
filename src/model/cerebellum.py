#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np

try:
    from .cerebrum import module_extended
except:
    from cerebrum import module_extended

class cerebellum(module_extended):
    r"""feedforward network which is an interpretation of the cerebellum
        paper: https://www.biorxiv.org/content/10.1101/2022.11.14.516257v1
    """
    def __init__(self, hidden_size, 
                 IO_sizes, IO_delays=None, IO_weights=None, num_hidden_layers=1, 
                 pons_sizes=None, nfibres=None, zero_output=False, 
                 bias=False, do_final_learning=False,rate_reg=0, nonlin='relu', 
                 fixed_MF=False, temp_basis=False, softmax=True):
        """
        Arguments
        ----------
        hidden_size : cerebellar hidden layer ('granule cells') size
        IO_sizes : sizes of different targets to the cerebellum
                    in paper, only task target is used and non-zero, but could have 
                    e.g. IO_sizes = {'targ': 4, 'PFC': 10}
                    this sets the total cerebellar target as a (4, 10 dimensional 
                    representation) the task target and RNN activity, respectively
        IO_delays : respective time windows (tau in paper) for different types of cerebelalr targets
        IO_weights : respective weight scaling for the loss of each type of target 
                    e.g. {'targ': 2, 'PFC': 1}) would place twice as much loss 
                    on the task target prediction. In paper only 'targ' is used,
                    so this argument is unused
        num_hidden_layers : number of cerebellar hidden layers
        pons_sizes : different pons (input to cerebellar cortex)
                    sizes to cerebellum; used to define input_size
        nfibres : number of connections ('mossy fibres') going into each GC, 
                  inspired by observed sparsity in cerebellum. Unused in paper
        zero_output : force (ablate) cerebellar output as zero
        bias : use bias in cerebellar weights
        do_final_learning : given > 0 IO_delays (time windows tau) train the
                           leftover activity on the (lastly available) targets
        rate_reg : rate regularisation on cerebellar activity (unused in paper)
        nonlin : granule cell non-linearity
        fixed_MF : fix mossy fibre weights during training
        temp_basis : use multiple cerebellar time windows (from given IO_size to 0)
                    'temporal basis' case in paper Fig 4
        softmax : for classification tasks, bound cerebellar predictions of task
                 target using softmax 
        """
        super(cerebellum, self).__init__()
        
        self.rate_reg = rate_reg
        self.fixed_MF = fixed_MF
        self.rate_reg = 0.
        if self.rate_reg > 0:
            self.rate_loss_fn = torch.nn.L1Loss()
        
        self.bias = bias
        self.temp_basis = temp_basis
        self.zero_output = zero_output
        self.record_losses = True #record cerebellar losses during training
        self.save_outputs = False
        self.do_final_learning = do_final_learning
        self.softmax = softmax
        
        self.noise = 0 #add noise to cerebellar output (see Fig S6 in paper)
        
        self.organise_inferior_olive(IO_sizes, IO_delays, IO_weights) #determines cerebellar targets
    
        self.input_size = sum(pons_sizes.values())
        self.output_size = sum(self.IO_sizes.values())
        
        if temp_basis:
            self.output_size *= (max(self.IO_delays.values()) + 1)

        self.hidden_size = hidden_size
        self.nfibres = nfibres
        self.num_hidden_layers = num_hidden_layers
        
        if nonlin == 'relu':
            self.f = torch.relu
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
        
        self.init_weights()
        if nfibres is not None and nfibres < self.input_size:
            self.control_mossy_fibres()
                
        self.init_targets()
        
        if self.record_losses:
            self.recorded_losses = {key: [] for key in self.IO_sizes.keys()}
            self.recorded_targets = []
            self.recorded_preds = []
            
        self.trained_inputs = []                
        self.targ_window = self.init_targ_window() #position of target indices in total cerebellar output 
                                                   # (e.g. cerebellum might predict both RNN + targets)
    
    def make_consistent(self, IO_sizes, IO_delays, IO_weights):
        r"""housekeeping to make sure all IO information has the same keys
        """
        for d in IO_delays, IO_weights:
            keys = list(d.keys())
            for key in keys:
                if key not in IO_sizes.keys():
                    d.pop(key)
    
    def organise_inferior_olive(self, IO_sizes, IO_delays, IO_weights):
        r""" housekeeping to make sure cerebellar target ('inferior olive')
            info is ok"""
        for name, dic in zip(['size', 'delay', 'weight'], [IO_sizes, IO_delays, IO_weights]):
            if dic is not None:
                suffix = '_IO_' + name
                for key in list(dic.keys()):
                    new_key = key.replace(suffix, '')
                    dic[new_key] = dic.pop(key)
        
        if IO_delays is None:
            IO_delays_all = {'PFC': 5, 'MC': 3, 'targ': 3, 'inp': 2} #example sizes
            IO_delays = dict.fromkeys(IO_sizes.keys(), 1)
            for key in IO_delays.keys():
                IO_delays[key] = IO_delays_all[key]

        if IO_weights is None:
            IO_weights = dict.fromkeys(IO_sizes.keys(), 1)
        
        self.make_consistent(IO_sizes, IO_delays, IO_weights)
        assert IO_delays.keys() == IO_sizes.keys(), 'require same IO keys for sizes/delays'
        assert IO_weights.keys() == IO_sizes.keys(), 'require same IO keys for sizes/weights'   
        
        self.IO_delays = IO_delays
        self.IO_sizes = IO_sizes
        self.IO_weights = IO_weights 
        
    def init_weights(self):
        top_layer_dim = self.output_size if self.num_hidden_layers == 0 else self.hidden_size
        
        self.input_trigger = torch.nn.Linear(
            in_features=self.input_size, out_features=top_layer_dim, bias=self.bias
        )
        
        layer_list = []
        for layer in range(self.num_hidden_layers):
            in_size = self.hidden_size
            out_size = self.hidden_size if layer < self.num_hidden_layers - 1 else self.output_size                
            layer = torch.nn.Linear(in_size, out_size, bias=self.bias)
            layer_list.append(layer)
                
        self.layers = torch.nn.ModuleList(layer_list)
                   
    def init_loss_fns(self):
        r"""cerebellar loss functions
        """
        self.loss = torch.nn.MSELoss()
        if self.classification:
            self.loss_targ = torch.nn.CrossEntropyLoss()
        else:          
            self.loss_targ = self.loss        

    def control_mossy_fibres(self):
        r"""sparsify input 'mossy fibre' weights of cerebellar network"""
        if self.nfibres is None:
            return
        else:           
            def generate_mask(input_shape, nfibres):
                mask = torch.ones(input_shape)
                for i in range(input_shape[0]):
                    connections = np.random.choice(input_shape[1], nfibres, replace=False)
                    mask[i, connections] = 0
                mask = mask.type(torch.bool)        
                return mask          
            
            def enforce_sparse(mask):
                def makezero(grad):
                    grad[mask] = 0          
                return makezero
            
            input_shape = self.input_trigger.weight.shape
            mask = generate_mask(input_shape, self.nfibres)
            with torch.no_grad():
                self.input_trigger.weight[mask] = 0
                print("Initialised sparse synthesiser weights")
            
            self.input_trigger.weight.register_hook(enforce_sparse(mask))
        
    def make_pred(self, input):            
        pred = self.input_trigger(input)
        
        if self.rate_reg > 0 or self.save_outputs:
            self.h = pred
        for layer in self.layers:         
            pred = layer(self.f(pred))
        
        return pred
        
    def forward(self, input, IO_info, final_learning=False):              
        pred = self.make_pred(input)
        
        if self.training:
            self.update_info(pred, IO_info)       
            self.learn()
            if final_learning:
                self.final_learning()
        
        if self.save_outputs:
            self.save_output(pred)

        new_pred = pred.detach().clone()

        if self.softmax and self.classification and self.targ_window is not None:
            new_pred[:, self.targ_window] = torch.softmax(new_pred[:, self.targ_window], dim=-1)

        if self.zero_output:
            new_pred = torch.zeros_like(new_pred)
        
        if self.noise > 0:
            new_pred += self.noise * torch.randn_like(new_pred)
        return new_pred
        
    def init_targets(self):
        max_IO_delay = max(self.IO_delays.values())
        pred_buffer_size = max_IO_delay + 1
        
        self.preds_buffer = [None] * pred_buffer_size
        
        self.targets = {}
        for target_type, target_delay in self.IO_delays.items():
            target_size =  max_IO_delay - target_delay + 1
            
            if self.temp_basis:
                target_size += target_delay
            self.targets[target_type] = [None] * target_size
        
        #if we're explicitly training with 0 delay, we care about the last timestep, otherwise we don't
        if max_IO_delay > 0:
            self.final_end = len(self.preds_buffer) - 1
        else:
            self.final_end = len(self.preds_buffer)  
         
    
    def update_info(self, pred, current_info):
        r"""update  predictions/target buffers
        """
        
        self.preds_buffer.pop(0)
        self.preds_buffer.append(pred)
        
        for key, target_list in self.targets.items():
            cinfo = current_info[key]              
            target_list.pop(0)
            
            if self.temp_basis:
                for i, target in enumerate(target_list):
                    if target is not None:
                        target = torch.cat((target, cinfo), dim=1)
                    else:
                        target = cinfo
                    
                    target_list[i] = target

            target_list.append(cinfo)


    def reset_buffers(self):
        if self.training:
            self.preds_buffer = [None] * len(self.preds_buffer)
            for key, targ_list in self.targets.items():
                self.targets[key] = [None] * len(targ_list)
    
    def learn(self):
        pred_whole = self.preds_buffer[0]
        if pred_whole is None:
            return
        preds_divided = self.divide_pred(pred_whole)        
        targets_divided = self.get_targets()
              
        loss = self.get_loss(preds_divided, targets_divided)

        if self.rate_reg > 0:
            loss += self.rate_reg * torch.norm(pred_whole)
            if not self.fixed_MF:
                loss += self.rate_reg * torch.norm(self.h)                
            
        if loss > 0:
            loss.backward()
        
        
    def final_learning(self):  
        r"""once the sequence is finished there may be things left in the buffer
        """
        if not self.do_final_learning:
            return
        for i in range(1, self.final_end):
            pred_whole = self.preds_buffer[i]
            if pred_whole is None:
                continue
            preds_divided = self.divide_pred(self.preds_buffer[i])
            targets_divided = self.get_targets(ind=i)
            
            loss = self.get_loss(preds_divided, targets_divided)
            
            if loss > 0:
                loss.backward()
    
    def divide_pred(self, pred):
        r"""divide cer. prediction in same way as IO modules
        """
        pred_divided = []
        b = 0
        for key, size in self.IO_sizes.items():
            if self.temp_basis:
                size *= (self.IO_delays[key] + 1)
            
            pred_divided.append(pred[:, b:b+size])
            b += size
        
        return tuple(pred_divided)

    def get_targets(self, ind=0):
        r"""get appropriate cerebellar targets
        """
        if ind == 0:
            return tuple([t[0] for t in self.targets.values()])
        else:     
            inds = [ind if ind < len(t) else -1 for t in self.targets.values()]
            return tuple([t[i] for t,i in zip(self.targets.values(), inds)])
    
    def get_loss(self, preds_divided, targets_divided):
        r"""cerebellar loss - the sum of all different 'module' losses
        e.g. PFC loss + input loss
        """
        loss = 0
            
        for i, (IO_type, IO_weight) in enumerate(self.IO_weights.items()):
            pred, target = preds_divided[i], targets_divided[i]
            
            loss_part = 0
            if target is not None:
                if pred.shape != target.shape:
                    if self.temp_basis:
                        target_size = target.shape[1]
                        pred = pred[:, :target_size]  
                    else:
                        assert self.classification, 'pred shape ({}) mismatched to target ({})'.format(pred.shape, target.shape)
                
                if 'targ' in IO_type:
                    loss_part = IO_weight * self.loss_targ(pred, target)
                else:
                    loss_part = IO_weight * self.loss(pred, target)
                loss += loss_part
            
            if self.record_losses and loss_part > 0:
                self.recorded_losses[IO_type].append(loss_part.item())   
                
                if not self.classification:
                    target_norm = torch.mean(torch.sum(target**2, 1))/2
                    self.recorded_targets.append(target_norm.item())
                    
                    pred_norm = torch.mean(torch.sum(pred**2, 1))/2
                    self.recorded_preds.append(pred_norm.detach().item())        
        return loss
    
    def force_zero_output(self, zero=True):
        r"""a slient/ablated cerebellum
        """
        self.zero_output = zero
    
    def record_outputs(self, record=True, seqlen=None):
        self.save_outputs = record
        if seqlen is None:
            seqlen = self.seqlen
        if record:
            self.n = 0
            shape_output = (seqlen, self.batch_size, self.output_size)
            self.recorded_outputs = torch.zeros(shape_output, requires_grad=False)

            shape_hidden = (seqlen, self.batch_size, self.hidden_size)
            self.recorded_hiddens = torch.zeros(shape_hidden, requires_grad=False)
    
    def save_output(self, pred):
        old_shape = self.recorded_outputs.shape
        if self.batch_size != old_shape[1]:
            new_shape = (old_shape[0], self.batch_size, old_shape[2])
            self.recorded_outputs = torch.zeros(new_shape, requires_grad=False)
            
            new_shape_hidd = (old_shape[0], self.batch_size, self.recorded_hiddens.shape[2])
            self.recorded_hiddens = torch.zeros(new_shape_hidd, requires_grad=False)
            
        if self.n >= old_shape[0]:
            print("Self.n is {} but old shape seqlen is {}".format(self.n, old_shape[0]))
            self.recorded_outputs = self.recorded_outputs.roll(-1, 0)
            self.recorded_hiddens = self.recorded_hiddens.roll(-1, 0)
            self.n -= 1

        self.recorded_outputs[self.n] = pred.detach()
        self.recorded_hiddens[self.n] = self.h.detach()
        self.n += 1
        
    def init_targ_window(self):
        r"""if cerebellum has multiple targets, work out which inds are the task target"""
        start = 0
        for key, size in self.IO_sizes.items():
            if key == 'targ':
                end = start + size
                return torch.arange(start, end)
            else:
                start += size
        return None
    