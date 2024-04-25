# delta regime functions
import numpy as np

def delta_0(delta, multiplier, epoch, tot_epoch=None):
    return 0

def delta_linear(delta, multiplier, epoch, tot_epoch=None): # linear growth
    return delta * multiplier * (epoch + 1)

def delta_exp(delta, multiplier, epoch, tot_epoch=None): # exponential growth of delta
    if epoch < 300: # set up limit to epoch values to avoid overflows
        delta_incr = delta * multiplier * np.exp(epoch + 1)
    else: 
        delta_incr = 1
    return delta_incr

def delta_sqrt(delta, multiplier, epoch, tot_epoch=None): # square root growth, with mult 0.7 it reaches 1 around 200 epochs
    return delta * multiplier * np.sqrt(epoch + 1)

def delta_square(delta, multiplier, epoch, tot_epoch=None): # square growth, use ~mult0.0002 to have 1 at 200 epochs
    return delta * multiplier * ((epoch + 1)**2)

def delta_mult_log(delta, multiplier, epoch, tot_epoch=None): # per avere 0.85 di delta a 100 epoch metti m~1.84, regime log da usare!
    return delta * multiplier * np.log(epoch + 2)
