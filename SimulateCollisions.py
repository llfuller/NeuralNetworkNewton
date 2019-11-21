import scipy as sp
from scipy.stats import uniform as uni
import time
import matplotlib.pyplot as plt

#   Creates elastic collision between two particles for arbitrary masses, center of mass velocities, and
# center of momentum angles with respect to horizontal. First generates 1D collision along x-axis, with
# particle 1 on +x-axis and particle 2 on -x-axis. The particles are rotated arbitrarily around the collision point,
# and then a center of mass velocity is added to both velocities. The frame in which the center of mass moves is
# the laboratory frame. All important physical quantity arrays are saved to LabValues.npz for later retrieval.
# Author: Lawson Fuller


start_time = time.time()
numTrials = 6000

# Notation:
# Letter "a" in variable name means quantity is in aligned (mass 1 on +x axis) CM frame
# Letter "n" is "not aligned"/unaligned frame
# Letter "L" is lab frame (CM velocity nonzero)

def energy(m1, m2, v1, v2):
    v1_norm_squared = sp.array([sp.power(v1[trial][0],2) + sp.power(v1[trial][1],2) for trial in range(len(v1))])
    v2_norm_squared = sp.array([sp.power(v2[trial][0],2) + sp.power(v2[trial][1],2) for trial in range(len(v2))])
    E1 = sp.multiply(0.5,  sp.multiply(m1,v1_norm_squared))
    E2 = sp.multiply(0.5,  sp.multiply(m2,v2_norm_squared))
    return sp.real(E1 + E2)
def x_momentum(m1, m2, v1, v2):
    v1_x = sp.array([v1[trial][0] for trial in range(len(v1))])
    v2_x = sp.array([v2[trial][0] for trial in range(len(v2))])
    p1 = sp.multiply(m1, v1_x)
    p2 = sp.multiply(m2, v2_x)
    return sp.real(p1 + p2)
def y_momentum(m1, m2, v1, v2):
    v1_y = sp.array([v1[trial][1] for trial in range(len(v1))])
    v2_y = sp.array([v2[trial][1] for trial in range(len(v2))])
    p1 = sp.multiply(m1, v1_y)
    p2 = sp.multiply(m2, v2_y)
    return sp.real(p1 + p2)
def conservation(a_i, a_f):
    # print("a_i: " +str(a_i[0]))
    # print("a_f: " +str(a_f[0]))
    # print("error: " +str(sp.fabs(sp.divide((a_i - a_f),a_i))[0]))
    return str( False not in (sp.isclose(a_i,a_f)))
def position(v_arr,t,pos_init):
    # v_arr is dimension (numTrials, 2)
    return None

#===========================================
#
# CM Aligned Frame
#
#===========================================
print("Beginning Aligned Center of Mass Phase: ")
print(str(time.time() - start_time)+" seconds")

# min and max of sampled values
m_min = 0.001
m_max = 10
x_min = 10
x_max = 100
v1ax_min = 0.001
v1ax_max = 10
vcm_min = -10
vcm_max = 10

# Random physical initial states for all trials with dimension (numTrials)
m1_Arr = uni.rvs(m_min, m_max, size=numTrials)
m2_Arr = uni.rvs(m_min, m_max, size=numTrials)
x1a_Arr = uni.rvs(x_min, x_max, size=numTrials)
v1ax_Arr = sp.multiply(-1,uni.rvs(v1ax_min, v1ax_max, size=numTrials))
vcm_x_Arr = uni.rvs(vcm_min, vcm_max, size=numTrials)
vcm_y_Arr = uni.rvs(vcm_min, vcm_max, size=numTrials)

# Always zero quantities with dimension (numTrials)
y1a_Arr = sp.zeros(numTrials)
y2a_Arr = sp.zeros(numTrials)
v1ay_Arr = sp.zeros(numTrials)
v2ay_Arr = sp.zeros(numTrials)

# Time until collision
t_col = sp.fabs(sp.divide(x1a_Arr,v1ax_Arr))
# Velocity of particle 2 with dimension (numTrials)
v2ax_Arr = sp.fabs(sp.multiply(v1ax_Arr,sp.divide(m1_Arr,m2_Arr)))
# Initial placement of particle 2 with dimension (numTrials)
x2a_Arr = sp.multiply(-1,sp.multiply(v2ax_Arr,t_col))


# Full aligned vectors with dimensions (2, numTrials)
v1a = sp.array([v1ax_Arr,v1ay_Arr])
v2a = sp.array([v2ax_Arr,v2ay_Arr])
InitialPosition1a = sp.array([x1a_Arr,y1a_Arr])
InitialPosition2a = sp.array([x2a_Arr,y2a_Arr])

# Initial energy in CM frame
E_i = energy(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))

# Final states (after collision) with vector dimension (numTrials):
m2Overm1Times2 = sp.multiply(2,sp.divide(m2_Arr,m1_Arr))
v1a_total_f =                sp.sqrt( sp.divide(sp.multiply(m2Overm1Times2,E_i),(m1_Arr+m2_Arr)) )
v2a_total_f = sp.multiply(-1,sp.multiply(sp.divide(m1_Arr,m2_Arr),v1a_total_f))
# Final x velocities in CM Frame with dimension (2, numTrials)
v1a_f = sp.array( [v1a_total_f, sp.zeros(numTrials)])
v2a_f = sp.array( [v2a_total_f, sp.zeros(numTrials)])

E_i = energy(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))
E_f = energy(m1_Arr,m2_Arr,sp.transpose(v1a_f),sp.transpose(v2a_f))
p_x_i = x_momentum(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))
p_y_i = y_momentum(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))
p_x_f = x_momentum(m1_Arr,m2_Arr,sp.transpose(v1a_f),sp.transpose(v2a_f))
p_y_f = y_momentum(m1_Arr,m2_Arr,sp.transpose(v1a_f),sp.transpose(v2a_f))

#===========================================
#
# CM Unaligned Frame
#
#===========================================
print("Beginning Rotation Phase")
print(str(time.time() - start_time)+" seconds")

# Rotation angles:
phi = sp.multiply(sp.random.uniform(0,1,size=numTrials),2*sp.pi) #remove the zero
# Rotation matrix with dimensions (2 , 2 , numTrials)
R = sp.array([[sp.cos(phi), sp.sin(phi)],[-sp.sin(phi), sp.cos(phi)]])

v1n = sp.empty(shape=(numTrials, 2))
v2n = sp.empty(shape=(numTrials, 2))
v1n_f = sp.empty(shape=(numTrials, 2))
v2n_f = sp.empty(shape=(numTrials, 2))
InitialPosition1n = sp.empty(shape=(numTrials, 2))
InitialPosition2n = sp.empty(shape=(numTrials, 2))
# Rotating initial vectors of particles 1 and 2 with dimensions (numTrials, 2)
for trial in range(numTrials):
    v1n[trial] = sp.matmul(R[:,:,trial], v1a[:,trial])
    v2n[trial] = sp.matmul(R[:,:,trial], v2a[:,trial])
    v1n_f[trial] = sp.matmul(R[:,:,trial], v1a_f[:,trial])
    v2n_f[trial] = sp.matmul(R[:,:,trial], v2a_f[:,trial])
    InitialPosition1n[trial] = sp.matmul(R[:,:,trial], InitialPosition1a[:,trial])
    InitialPosition2n[trial] = sp.matmul(R[:,:,trial], InitialPosition2a[:,trial])

#===========================================
#
# Lab Frame Velocities
#
#===========================================
print("Beginning Lab Phase: ")
print(str(time.time() - start_time)+" seconds")

# Full center of mass velocity in lab frame. Dimension (numTrials , 2)
vcm = sp.transpose(sp.array([vcm_x_Arr, vcm_y_Arr]))

# Particle velocities in  lab frame. Dimension (numTrials, 2)
v1L = v1n + vcm
v2L = v2n + vcm
v1L_f = v1n_f + vcm
v2L_f = v2n_f + vcm

# Initial positions in lab frame:
InitialPosition1L = InitialPosition1n
InitialPosition2L = InitialPosition2n

#===========================================
#
# Lab Frame Conserved Quantities
#
#===========================================
E_i = energy(m1_Arr,m2_Arr,v1L,v2L)
E_f = energy(m1_Arr,m2_Arr,v1L_f,v2L_f)
p_x_i = x_momentum(m1_Arr,m2_Arr,v1L,v2L)
p_y_i = y_momentum(m1_Arr,m2_Arr,v1L,v2L)
p_x_f = x_momentum(m1_Arr,m2_Arr,v1L_f,v2L_f)
p_y_f = y_momentum(m1_Arr,m2_Arr,v1L_f,v2L_f)

#===========================================
#
# Lab Frame Collision Time Series
#
#===========================================
print("Beginning Path Phase: ")
print(str(time.time() - start_time)+" seconds")

timeStepOfCollision = 100
timeSteps = 2*timeStepOfCollision - 1
# Dimension (timeSteps)
t = sp.linspace(0, sp.multiply(2,t_col), timeSteps)

# Rework dimensions of initial position and velocity matrices during this step

# Dimension (2(the # of SpatialDimensions), numTrials, timeSteps)
Position1L_t_pre = sp.repeat(sp.transpose(InitialPosition1L)[:,:,sp.newaxis], timeSteps, axis=2)\
               + sp.multiply( sp.repeat(sp.transpose(v1L)[:, :, sp.newaxis], timeSteps, axis=2), sp.transpose(t) )
Position2L_t_pre = sp.repeat(sp.transpose(InitialPosition2L)[:,:,sp.newaxis], timeSteps, axis=2) \
               + sp.multiply( sp.repeat(sp.transpose(v2L)[:, :, sp.newaxis], timeSteps, axis=2), sp.transpose(t) )
Position1L_t_post = sp.repeat(Position1L_t_pre[:,:,timeStepOfCollision][:,:,sp.newaxis], timeSteps, axis=2) \
               + sp.multiply( sp.repeat(sp.transpose(v1L_f)[:, :, sp.newaxis], timeSteps, axis=2), sp.transpose(t) )
Position2L_t_post = sp.repeat(Position2L_t_pre[:,:,timeStepOfCollision][:,:,sp.newaxis], timeSteps, axis=2) \
               + sp.multiply( sp.repeat(sp.transpose(v2L_f)[:, :, sp.newaxis], timeSteps, axis=2), sp.transpose(t) )
Position1L_t = sp.concatenate((Position1L_t_pre[:,:,:timeStepOfCollision],Position1L_t_post[:,:,timeStepOfCollision:]),2)
Position2L_t = sp.concatenate((Position2L_t_pre[:,:,:timeStepOfCollision],Position2L_t_post[:,:,timeStepOfCollision:]),2)
Velocity1L_t_pre = sp.repeat(sp.transpose(v1L)[:, :, sp.newaxis], timeSteps, axis=2)
Velocity2L_t_pre = sp.repeat(sp.transpose(v2L)[:, :, sp.newaxis], timeSteps, axis=2)
Velocity1L_t_post = sp.repeat(sp.transpose(v1L_f)[:, :, sp.newaxis], timeSteps, axis=2)
Velocity2L_t_post = sp.repeat(sp.transpose(v2L_f)[:, :, sp.newaxis], timeSteps, axis=2)
Velocity1L_t = sp.concatenate((Velocity1L_t_pre[:,:,:timeStepOfCollision],Velocity1L_t_pre[:,:,timeStepOfCollision:]),2)
Velocity2L_t = sp.concatenate((Velocity2L_t_pre[:,:,:timeStepOfCollision],Velocity2L_t_pre[:,:,timeStepOfCollision:]),2)

# Save arrays to file
print("Saving: ")
print(str(time.time() - start_time)+" seconds")
sp.savez("LabValues", v1L, v2L, v1L_f, v2L_f, E_i, E_f, p_x_i, p_x_f, p_y_i, p_y_f,
         t, Position1L_t, Position2L_t, Velocity1L_t, Velocity2L_t)
print("Time to run: ")
print(str(time.time() - start_time)+" seconds")
