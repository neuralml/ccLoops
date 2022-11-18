import sys
from sacred import Ingredient

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parent_parentdir  = os.path.dirname(parentdir)
src_dir = os.path.join(parent_parentdir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.handlers import Tracer, ModelCheckpoint
from training.loss import init_metrics
from training.optimizer import init_optimizer, init_lr_scheduler
from training.engine import create_rnn_trainer, create_rnn_evaluator

training = Ingredient('training')

init_metrics = training.capture(init_metrics)
init_optimizer = training.capture(init_optimizer)
create_rnn_trainer = training.capture(create_rnn_trainer)
create_rnn_evaluator = training.capture(create_rnn_evaluator)

init_lr_scheduler = training.capture(init_lr_scheduler)
