import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def init_optimizer(optimizer, model, lr_rnn=0.001, lr_readout=0.001, lr_cerebellum=0.001, l2_norm=0.0, **kwargs):

    params = [{'params': model.cerebrum.rnn.parameters(), 'weight_decay': l2_norm, 'lr': lr_rnn},
                  {'params': model.cerebrum.readout.parameters(), 'lr': lr_readout, 'weight_decay': l2_norm}]
   
    if model.apply_cerebellum:
        params.append({'params': model.cerebellum.parameters(), 'lr': lr_cerebellum, 'weight_decay': l2_norm
                       })
        if hasattr(model, 'auto_encoder'):
            params.append({'params': model.auto_encoder.parameters()})

    if optimizer == 'adam':
        optimizer = optim.Adam(params, eps=1e-9, betas=[0.9, 0.98])
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(params,
            lr=lr_rnn, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(params,
            lr=lr_rnn, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params,
            lr=lr_rnn, eps=1e-10, weight_decay=l2_norm, momentum=0.9)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params,
            lr=lr_rnn, weight_decay=l2_norm, momentum=0.9)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
            lr=lr_rnn, weight_decay=l2_norm, lr_decay=0.9)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(params,
            lr=lr_rnn, weight_decay=l2_norm, rho=0.9)
    else:
        raise ValueError(r'Optimizer {0} not recognized'.format(optimizer))

    return optimizer


def init_lr_scheduler(optimizer, scheduler, lr_scale, lr_decay_patience, threshold=1e-4, min_lr=1e-9):

    if scheduler == 'reduce-on-plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_scale,
            patience=lr_decay_patience,
            threshold=threshold,
            min_lr=min_lr
        )
    elif scheduler == 'cycle':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1, 
            epochs=100, 
            steps_per_epoch=100)
    else:
        raise ValueError(r'Scheduler {0} not recognized'.format(scheduler))

    return scheduler