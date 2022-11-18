import torch
import torch.nn as nn

from ignite.engine import Events, Engine, _prepare_batch

def _detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """
    if hidden_state is None:
        return None
    elif isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [_detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(_detach_hidden_state(h) for h in hidden_state)
    raise ValueError('Unrecognized hidden state type {}'.format(type(hidden_state)))

########################################################################################
# Training
########################################################################################
def create_rnn_trainer(model, optimizer, loss_fn, grad_clip=0, reset_hidden=True,
                    device=None, non_blocking=False, prepare_batch=_prepare_batch):
    
    if device:
        model.to(device)

    def _training_loop(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()
        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
        hidden = engine.state.hidden
        pred, hidden, targets = model(inputs, hidden=hidden, target=targets)
                        
        loss = loss_fn((pred, hidden[0]), targets)
            
        loss.backward()
                        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  
        
        # Optimize
        optimizer.step()
        
        if not reset_hidden:
            engine.state.hidden = hidden
            
        return loss.item()


    # If reusing hidden states, detach them from the computation graph
    # of the previous batch. Usin the previous value may speed up training
    # but detaching is needed to avoid backprogating to the start of training.
    def _detach_wrapper(engine):
        if not reset_hidden:
            engine.state.hidden = _detach_hidden_state(engine.state.hidden)
    
    loop = _training_loop
    
    engine = Engine(loop)
    engine.add_event_handler(Events.EPOCH_STARTED, lambda e: setattr(e.state, 'hidden', None))
    engine.add_event_handler(Events.ITERATION_STARTED, _detach_wrapper)

    return engine

########################################################################################
# Validation
########################################################################################
def create_rnn_evaluator(model, metrics, device=None, hidden=None, non_blocking=False,
                        prepare_batch=_prepare_batch,):
    if device:
        model.to(device)
        
    def _inference(engine, batch):
        model.eval()

        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            pred, _ = model(inputs)
            
        if model.classification and not model.predict_last and model.final_window is None:
            pred = pred.permute((0, 2, 1))
            
        if model.final_window:
            targets = targets[:, -1]
                        
        return pred, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def run_training(
        model, train_data, trainer, epochs,
        metrics, test_data, model_checkpoint, device
    ):
    trainer.run(train_data, max_epochs=epochs)
    
    # Select best model
    best_model_path = model_checkpoint.last_checkpoint
    with open(best_model_path, mode='rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    tester = create_rnn_evaluator(model, metrics, device=device)
    tester.run(test_data)
    
    if hasattr(trainer.state, 'tgrads'):
        grads = (trainer.state.hidds, trainer.state.truegrads, trainer.state.tgrads, trainer.state.sgrads)
        return tester.state.metrics, grads
    else:    
        return tester.state.metrics