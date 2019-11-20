import scipy as sp
import matplotlib as plt

numTrials = 1000

# min and max of sampled values
m_min = 0.001
m_max = 10
x_min = 10
x_max = 100
v1ax_min = -0.001
v1ax_max = -10
vcm_min = -50
v_cm_max = 50

# random arrays of floats between 0 and 1 used for all trials
rand_for_m1 = sp.random(numTrials)
rand_for_m2 = sp.random(numTrials)
rand_for_x1a = sp.random(numTrials)
rand_for_x2a = sp.random(numTrials)
rand_for_v1ax = sp.random(numTrials)
rand_for_vcm_x = sp.random(numTrials)
rand_for_vcm_y = sp.random(numTrials)

# Arrays of masses for all trials

