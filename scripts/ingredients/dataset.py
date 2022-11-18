import sys
from sacred import Ingredient
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parent_parentdir  = os.path.dirname(parentdir)
src_dir = os.path.join(parent_parentdir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    

from dataset.delayed_association import load_delayed_association
from dataset.linedraw import load_linedraw
from dataset.digit_draw import load_digit_draw
from dataset.evidence_accumulation import load_evidence_accumulation

linedraw = Ingredient('dataset')
load_linedraw = linedraw.capture(load_linedraw)

digit_draw = Ingredient('dataset')
load_digit_draw = digit_draw.capture(load_digit_draw)

evidence_accumulation = Ingredient('dataset')
load_evidence_accumulation = evidence_accumulation.capture(load_evidence_accumulation)

delayed_association = Ingredient('dataset')
load_delayed_association = delayed_association.capture(load_delayed_association)



