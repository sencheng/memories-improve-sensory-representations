# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:52:43 2016

@author: richaopf
"""

import random, copy

def choose(thing):
    chosen = []
        
    if type(thing) is dict:
        for key in sorted(thing.keys()):
            if type(key) is str and key.startswith('!'):
                options = thing.pop(key)
                index = random.randint(0, len(options)-1)
                thing[key[1:]] = options[index]
                chosen.append(index)  
            else:
                chosen.append(choose(thing[key]))
                 
    elif hasattr(thing, '__dict__'):
        return choose(thing.__dict__)
    
    elif type(thing) is list:
        for item in thing:
            chosen.append(choose(item))
        
    return thing, chosen
                 

def select(thing, to_choose):
    if type(thing) is dict:
        for key in sorted(thing.keys()):
            if type(key) is str and key.startswith('!'):
                options = thing.pop(key)
                index = to_choose.pop(0)
                thing[key[1:]] = options[index]
            else:
                select(thing[key], to_choose)                 
                 
    elif hasattr(thing, '__dict__'):
        select(thing.__dict__, to_choose)
    
    elif type(thing) is list:
        for item in thing:
            select(item, to_choose)
            
    return thing
            

def select_uniform(thing, index):
    if type(thing) is dict:
        for key in sorted(thing.keys()):
            if type(key) is str and key.startswith('!'):
                options = thing.pop(key)
                thing[key[1:]] = options[index % len(options)]
            else:
                select_uniform(thing[key], index)                 
                 
    elif hasattr(thing, '__dict__'):
        select_uniform(thing.__dict__, index)
    
    elif type(thing) is list:
        for item in thing:
            select_uniform(item, index)
            
    return thing