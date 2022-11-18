#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
multi-task training + switching (cf Fig 3 in paper)
"""

import sys, os
import numpy as np
import torch
import ignite
from ignite.engine import Events
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
import inspect

import matplotlib.pyplot as plt


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir)

# Load experiment ingredients
from ingredients.dataset import linedraw as dataset, load_linedraw as load_dataset
    
from ingredients.model import model, init_model
from ingredients.training import training, init_metrics, init_optimizer, \
                                 create_rnn_trainer, create_rnn_evaluator, \
                                 Tracer, ModelCheckpoint


import logging
logging.getLogger("ignite").setLevel(logging.WARNING)

# Add configs
training.add_config('configs/training.yaml')
model.add_config('configs/model.yaml')
dataset.add_config('configs/dataset.yaml')

OBSERVE = False #save model results
nepochs = 50
seed = 123

temp_path = 'sims/temp'

# Set up experiment    
ex_name = 'linedraw'
ex = Experiment(name=ex_name, ingredients=[dataset, model, training])
ex.add_config(no_cuda=False, save_folder=temp_path,
              experiment_name=ex_name)

ex.add_package_dependency('torch', torch.__version__)

if OBSERVE:
    ex.observers.append(FileStorageObserver.create('sims'))

# Functions
@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

#@ex.automain
@ex.main
def main(_config, seed):
    no_cuda = _config['no_cuda']
    epochs = _config['training']['epochs']
        
    input_size = _config['dataset']['input_D']
    output_size = 2
    seq_len =  _config['dataset']['seq_len']
    classification = False      
    predict_last = False     
    
    log_interval = max(1, epochs // 10)
    log_interval = 5
    
    batch_size = _config['training']['batch_size']

    # Init metrics
    loss, metrics = init_metrics('mse', ['mse'])

    device = set_seed_and_device(seed, no_cuda)
    
    training_set, validation_set, test_set = load_dataset(batch_size=batch_size)
    print("retrieved dataset")  
    model = init_model(input_size=input_size, output_size=output_size)
    print("initialised model")    

    model.init_dataset_details(batch_size, seq_len, classification,
                                 predict_last)    
        
    model = model.to(device=device)
    model.set_2task_model() #multi-task learning
    
    optimizer = init_optimizer(model=model) 
    
    # Init engines
    trainer = create_rnn_trainer(model, optimizer, loss, device=device)
    validator = create_rnn_evaluator(model, metrics, device=device)

    @trainer.on(Events.EPOCH_STARTED)
    def print_epoch(engine):
        nepoch = engine.state.epoch
        if nepoch % log_interval == 0:
            print('#'*75)
            print("Epoch: {}".format(engine.state.epoch))   
            print('#'*75)
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_set)
    
    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.should_terminate=True

    # Record training progression
    global tracer
    tracer = Tracer(metrics).attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):        
        loss_epoch = tracer.loss[-1]
        ex.log_scalar('training_loss', loss_epoch)
        tracer.loss.clear()
        
        if model.apply_cerebellum and model.cerebellum.record_losses:
            for IO_type, error_list in model.cerebellum.recorded_losses.items():
                name = 'cerebellar_{}_loss'.format(IO_type)
                loss_epoch = np.nanmean(error_list)
                ex.log_scalar(name, loss_epoch)
                
                error_list.clear()
                    
    val_metrics = {}
    for key in metrics.keys():
        val_metrics[key] = []

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            if 'mse' in metric: #correct for difference between torch.lossMSE and ignite.lossMSE
                if not model.predict_last:
                    value = value/seq_len 
                if metric == 'mse':
                    value = value/output_size
            
            print("Val loss:", value)
            ex.log_scalar('val_{}'.format(metric), value)
            val_metrics[metric].append(value)

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    global checkpoint
    checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='disent',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    
    validator.add_event_handler(Events.COMPLETED, checkpoint, {'model': model})
    
    print("starting training for task 1")
    trainer.run(training_set, max_epochs=epochs)
    print("finished training for task 1")
            
    #save final model (not necessarily best)
    final_model_path = temp_path + 'final-model.pt'
    torch.save(model.state_dict(), final_model_path)
    ex.add_artifact(final_model_path, 'final-model.pt')
    os.remove(final_model_path)
        
    # Run on test data
    model.eval()
    model.load_state_dict(checkpoint.best_model)
    tester = create_rnn_evaluator(model, metrics, device=device,)

    test_metrics = tester.run(test_set).metrics
    
    # Save best model performance and state
    for metric, value in test_metrics.items():
        print("Test {}: {}".format(metric, value))
        ex.log_scalar('test_{}'.format(metric), value)
    #ex.add_artifact(checkpoint.last_checkpoint, 'trained-model')
    ex.add_artifact(checkpoint.last_checkpoint, 'best-model.pt')
    
    os.remove(checkpoint.last_checkpoint)   
    
    model.eval()
    task1_mse_pre = tester.run(test_set).metrics['mse']
    data_task1, targets_task1 = next(iter(test_set))
    pred_task1_baseline, _ = model(data_task1)
        
    #consider new task with distinct cerebellar weights 
    pf_overlap = None #ratio of same PFs to use across tasks (None is zero)
        
    print("cereb parms at end of baseline:", model.cerebellum.input_trigger.weight[0,:5])
    model.set_task_number(1, pf_overlap=pf_overlap) 
    
    trainer.remove_event_handler(validate, Events.EPOCH_COMPLETED)
    
    # need to make sure the optimiser is updating the correct cerebellar weights
    del optimizer.param_groups[2]
    optimizer.add_param_group(
        {'params' : model.cerebellum.parameters()}
        )     
    
    training_set2, validation_set2, test_set2 = load_dataset(arc=1, 
                                                            batch_size=batch_size,)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_set2)
    
    
    trainer.state.max_epochs = None
    nepochs_task2 = 10
    print("starting training for task 2")
    trainer.run(training_set2, max_epochs=nepochs_task2)
    print("finished training for task 2")
            
    model.eval()
    task2_mse = tester.run(test_set2).metrics ['mse']   
    data_task2, targets_task2 = next(iter(test_set2))
    pred_task2, _ = model(data_task2)    
    
    model.set_task_number(0)
    
    task1_mse_post = tester.run(test_set).metrics['mse']
    pred_task1_switch, _ = model(data_task1)
    
    
    fig = plt.figure(0)
    x = np.arange(3)
    errors = [task1_mse_pre, task2_mse, task1_mse_post]
    xticklabels = ['task 1 (baseline)', 'task 2', 'task 1 (zero-shot switch)']
    plt.bar(x, errors)
    plt.xticks(x, labels=xticklabels)
    plt.ylabel("error (MSE)")
    
    fig = plt.figure(1, (6, 2))
    preds_all = [pred_task1_baseline, pred_task2, pred_task1_switch]
    targets_all = [targets_task1, targets_task2, targets_task1]
    
    for i, (preds, targets) in enumerate(zip(preds_all, targets_all)):
        plt.subplot(1, 3, i+1)
        plt.axis('off')
        plt.title(xticklabels[i])
        
        preds = preds.detach().numpy()
        targets = targets.numpy()
        unique_inds =  np.unique(targets, axis=0, return_index=True)[1]
        
        for ind in unique_inds:
            plt.plot(preds[ind, :, 0], preds[ind, :, 1], color='orange')
            plt.plot(targets[ind, :, 0], targets[ind, :, 1], color='black', linestyle='dotted')
            
        plt.xlim([-1,1])
        plt.ylim([-1,1])
    
    
if __name__ == "__main__":
    r = ex.run(config_updates={'training': {'epochs': nepochs},
               'model': {'apply_cerebellum': True,},
               'seed': seed})
