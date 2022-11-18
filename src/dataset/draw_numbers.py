#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
digit templates for digit-drawing task 
"""

import numpy as np

try:
    from .linedraw import draw_2dline
except:
    from linedraw import draw_2dline


def fit_points(seq_len, points):
    npairs = points.shape[0] - 1
    assert seq_len >= len(points), "Something went wrong"
    
    div = seq_len // npairs
    r = seq_len % npairs
    
    
    lengths = np.ones(npairs, dtype=int) * div
    
    if r > 0:
        lengths[:r-1] += 1 #black magic
    else:
        lengths[-1] -= 1
    

    lines = []    
    for i in range(npairs): #last pair is weird, deal with separately
        start_point = points[i]
        end_point = points[i + 1]
        line = draw_2dline(lengths[i] + 1, end_point, start_point)[:-1] #let next point be represented in the next pair
        lines.append(line)
    
    lines.append(points[-1].reshape(1, 2))
    
    lines = np.vstack(lines)
    
    assert lines.shape == (seq_len, 2), "Something went wrong"
    
    return lines    


#start from [0,1] and go counter clockwise
def draw0(seq_len):    
    x = np.array([0, 0, 0, 0.5, 1, 1, 1, 0.5, 0])
    y = np.array([1, 0.5, 0, 0, 0, 0.5, 1, 1, 1])
    points = np.vstack((x,y)).T
    
    curve = fit_points(seq_len, points)
    return curve
        
def draw1(seq_len):
    start_point = np.array([0.5, 1])
    end_point = np.array([0.5, 0])
    line = draw_2dline(seq_len, end_point, start_point)
    return line

def draw2(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0, 0.5, 1])
    y = np.array([1, 1, 1, 0.5, 0, 0,0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve   

def draw3(seq_len):
    x = np.array([0, 0.5, 1, 1, 0.5, 1, 1, 0.5, 0])
    y = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0, 0,0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve    

def draw4(seq_len):
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 1])
    y = np.array([1, 0.5, 0, 0.5, 1, 0.5, 0.5, 0.5])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve       

def draw5(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0, 0, 0.5, 1, 1, 0.5, 0])
    y = np.array([1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve 

def draw6(seq_len):
    x = np.array([0, 0, 0, 0.5, 1, 1, 0.5, 0])
    y = np.array([1, 0.5, 0, 0, 0, 0.5, 0.5, 0.5])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve

def draw7(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0])
    y = np.array([1, 1, 1, 0.5, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve

def draw8(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0])
    y = np.array([1, 0.5, 0, 0, 0, 0.5, 1, 1, 1])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve  

def draw9(seq_len):
    x = np.array([0, 0, 0.5, 0.5, 0, 0.5, 0.5, 0.5])
    y = np.array([1, 0.5, 0.5, 1, 1, 1, 0.5, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve   

def draw_number(number, seqlen):
    methodname = "draw" + str(number)
    drawdigit = globals()[methodname]
    digit = drawdigit(seqlen)    
    return digit