import sys
from sacred import Ingredient
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parent_parentdir  = os.path.dirname(parentdir)
src_dir = os.path.join(parent_parentdir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.cerebro_cerebellum import cerebro_cerebellum as init_model

model = Ingredient('model')
init_model = model.capture(init_model)
