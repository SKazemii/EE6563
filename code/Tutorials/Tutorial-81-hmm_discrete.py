# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:40:20 2021

@author: pkumar1
"""

states = ('Rainy', 'Sunny')
 
observations = ('walk', 'shop', 'clean')
 
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
transition_probability = {
   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
   }
 
emission_probability = {
   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
   }

from hmmlearn import hmm
import numpy as np

model = hmm.MultinomialHMM(n_components=2)
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3],
                            [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                [0.6, 0.3, 0.1]])

#............................................................
##Problem 1:Given a known model what is the likelihood of sequence O happening?
#...........................................................
import math

math.exp(model.score(np.array([[0]])))
math.exp(model.score(np.array([[1]])))
math.exp(model.score(np.array([[2]])))
math.exp(model.score(np.array([[1,1,2,2,2]])))

#..................................................................
#problem 2:Given a known model and sequence O, what is the optimal hidden state sequence?
#........................................................

logprob, seq = model.decode(np.array([[1,2,0]]).transpose())
print(math.exp(logprob))#probability measure
print(seq)

logprob, seq = model.decode(np.array([[2,2,2]]).transpose())
print(math.exp(logprob))
print(seq)

