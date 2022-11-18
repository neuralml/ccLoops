#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
cerebellar-to-cortical consolidation for delayed association task (cf Fig 8 in paper)
"""
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import copy

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parent_parentdir  = os.path.dirname(parentdir)
src_dir = os.path.join(parent_parentdir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

from dataset.delayed_association import load_delayed_association    
from model.CC_sleep import cc_sleep
    
results_path = 'sims/1/' #path to (pretrained) delayed association model

model_fn = 'best-model.pt'
config_fn = results_path + 'config.json'

with open(config_fn) as json_file:
    config = json.load(json_file)
dataset_config = config['dataset']
dataset_config['training_size'] = 10000 #just iterate over one big training session

#model params
input_size = dataset_config['input_D']
output_size = dataset_config['npoints']

#training params
nbatch = 50
batch_size = 10
decay_rate = 0.1
h_power = 1 #for bio learning rule
least_squares = True #optimal solution

if least_squares:
    cereb_neo_learning_ratio = 1
else:
    cereb_neo_learning_ratio = 13/1.5 #normalize by 13 timesteps where cerebellar provides (non-zero) predictions

optimizer = 'adam'

model = cc_sleep(config_fn, input_size, output_size,
                 decay_rate=decay_rate, cereb_neo_learning_ratio=cereb_neo_learning_ratio,
                 h_power=h_power, least_squares=least_squares)
training_set, validation_set, test_set = load_delayed_association(batch_size=batch_size, **dataset_config,
                                                                  )

model.init_dataset_details(batch_size, dataset_config['seq_len'], classification=True,
                             predict_last=True, final_window=dataset_config['window_size']) 

model.load_state_dict(torch.load(results_path+model_fn, map_location=torch.device('cpu')))
model.store_orig_weights()


def get_acc(pred):
    ncorrect = torch.sum(torch.argmax(pred, 1) == targets_test[:, 0])   
    acc = ncorrect/len(targets_test) 
    return acc

def test_ablation(model):
    model_ablation = copy.deepcopy(model)
    model_ablation._sleep_mode(False)
    pred_control, _ = model_ablation(data_test)
    
    model_ablation.cerebellum.force_zero_output()
    pred_ablation, _ = model_ablation(data_test)

    control_acc = get_acc(pred_control)
    ablation_acc = get_acc(pred_ablation)
    
    return control_acc, ablation_acc


data_test, targets_test = next(iter(test_set))
model._sleep_mode(False)


model.eval()
initial_ablation = test_ablation(model)
pred_base, _ = model(data_test, return_whole=True)

model._sleep_mode(True)


cc_norms = [] #cerebellar-cortical weight norms
accs = []
ablation_accs = []
partial_ablation_accs = []
for i, (data, target) in enumerate(training_set):
    if i >= nbatch:
        break
    
    if i % 5 == 0:
        print("consolidation trial {}/{}".format(i, nbatch))
    acc, ablation_acc = test_ablation(model)
    accs.append(acc)
    ablation_accs.append(ablation_acc)
    
    cc_norm = torch.norm(model.cerebrum.rnn.w_ih[:, input_size:])
    cc_norms.append(cc_norm)
    
    pred, hidden = model(data)   


model._sleep_mode(False)
pred_final, _ = model(data_test, return_whole=True)


model.cerebellum.force_zero_output()
pred_ablated, _ = model(data_test, return_whole=True)

pred_base = pred_base.detach().numpy()
pred_final = pred_final.detach().numpy()
pred_ablated = pred_ablated.detach().numpy()

plt.figure(0, (9, 8))
plt.subplot(221)
plt.plot(accs, label='control', color='orange')
plt.plot(ablation_accs, label='cerebellar ablation', color='purple')
plt.xlabel("sleep session")
plt.ylabel("accuracy")
plt.legend()

plt.subplot(222)
plt.plot(cc_norms, color='orange')
plt.xlabel("sleep session")
plt.ylabel("||cerebellar-cortico weight||")

sleep_errors = np.array(model.sleeping_errors)
m = sleep_errors.reshape(nbatch, -1)
m = np.mean(m, axis=1)


plt.subplot(223)
plt.plot(m)
plt.xlabel("sleep session")
plt.ylabel("cossim(old cc network, new cortex-only network)")

ind = 5 #example index
plt.subplot(224)
ex_pred_orig = pred_base[ind]
plt.plot(ex_pred_orig[:, 0], c='red', linestyle='--', label='control (pre sleep)')
plt.plot(ex_pred_orig[:, 1], c='green', linestyle='--')

ex_pred_final = pred_ablated[ind]
plt.plot(ex_pred_final[:, 0], c='red', label='ablation (post sleep)')
plt.plot(ex_pred_final[:, 1], c='green')
plt.xlabel("timestep")
plt.ylabel("selectivity")
plt.legend()

