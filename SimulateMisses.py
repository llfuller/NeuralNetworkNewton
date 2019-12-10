import scipy as sp
from scipy.stats import uniform as uni
import CalculateConserved as CalcConsd
import time
import matplotlib.pyplot as plt

#   Creates missed collision between two particles for arbitrary masses, center of mass velocities, and
# center of momentum angles with respect to horizontal.


start_time = time.time()
numTrials = 100000

# Notation:
# Letter "L" is lab frame (CM velocity nonzero)

#===========================================
#
# CM Frame
#
#===========================================
print("Beginning Aligned Center of Mass Phase: ")
print(str(time.time() - start_time)+" seconds")

# min and max of sampled values
m_min = 0.01
m_max = 6
r_min = 1
r_max = 1.1
var_min = 1
var_max = 5
vcm_min = -10
vcm_max = 10
timeSteps = 30
r = 1.0 # radius of each mass
radius_Arr= sp.multiply(sp.ones(numTrials), r)

for trainOrIntermediateFile in [0,1]:

    # Random physical initial states for all trials with dimension (numTrials)
    m1_Arr = uni.rvs(m_min, m_max, size=numTrials)
    m2_Arr = uni.rvs(m_min, m_max, size=numTrials)
    r1a_Arr = uni.rvs(r_min, r_max, size=numTrials)
    r2a_Arr = uni.rvs(r_min, r_max, size=numTrials)
    v1ar_Arr = sp.multiply(-1,uni.rvs(var_min, var_max, size=numTrials))
    v2ar_Arr = sp.multiply(-1,uni.rvs(var_min, var_max, size=numTrials))
    vcm_x_Arr = uni.rvs(vcm_min, vcm_max, size=numTrials)
    vcm_y_Arr = uni.rvs(vcm_min, vcm_max, size=numTrials)

    # Rotation angles:
    phi_Pos1 = sp.multiply(sp.random.uniform(0,1,size=numTrials),2*sp.pi) #remove the zero
    phi_Pos2 = sp.multiply(sp.random.uniform(0,1,size=numTrials),2*sp.pi) #remove the zero
    phi_Vel1 = sp.multiply(sp.random.uniform(0,1,size=numTrials),2*sp.pi) #remove the zero
    phi_Vel2 = sp.multiply(sp.random.uniform(0,1,size=numTrials),2*sp.pi) #remove the zero

    # Calculating velocities and positions of particles from radial components and generated phi
    x1a_Arr = sp.multiply (r1a_Arr, sp.cos(phi_Pos1))
    y1a_Arr = sp.multiply (r1a_Arr, sp.sin(phi_Pos1))
    x2a_Arr = sp.multiply (r2a_Arr, sp.cos(phi_Pos2))
    y2a_Arr = sp.multiply (r2a_Arr, sp.sin(phi_Pos2))
    v1ax_Arr = sp.multiply (v1ar_Arr, sp.cos(phi_Vel1))
    v1ay_Arr = sp.multiply (v1ar_Arr, sp.sin(phi_Vel1))
    v2ax_Arr = sp.multiply (v2ar_Arr, sp.cos(phi_Vel2))
    v2ay_Arr = sp.multiply (v2ar_Arr, sp.sin(phi_Vel2))

    # Full aligned vectors with dimensions (2, numTrials)
    v1a = sp.array([v1ax_Arr,v1ay_Arr])
    v2a = sp.array([v2ax_Arr,v2ay_Arr])
    InitialPosition1a = sp.array([x1a_Arr,y1a_Arr])
    InitialPosition2a = sp.array([x2a_Arr,y2a_Arr])

    # Initial energy in CM frame
    E_i = CalcConsd.energy(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))

    # Final states (after collision) with vector dimension (numTrials):
    m2Overm1Times2 = sp.multiply(2,sp.divide(m2_Arr,m1_Arr))

    E_i = CalcConsd.energy(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))
    p_x_i = CalcConsd.x_momentum(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))
    p_y_i = CalcConsd.y_momentum(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))

    #===========================================
    #
    # CM Unaligned Frame
    #
    #===========================================
    print("Beginning Rotation Phase")
    print(str(time.time() - start_time)+" seconds")

    # Rotation angles:
    phi = sp.zeros((numTrials)) #remove the zero
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
        v1n[trial] = 0*sp.matmul(R[:,:,trial], v1a[:,trial])
        v2n[trial] = 0*sp.matmul(R[:,:,trial], v2a[:,trial])
        InitialPosition1n[trial] = sp.matmul(R[:,:,trial], InitialPosition1a[:,trial])
        InitialPosition2n[trial] = sp.matmul(R[:,:,trial], InitialPosition2a[:,trial])

    #===========================================
    #
    # Mark as collision or miss
    #
    #===========================================
    # Time of nearest approach or minimum distance: (dim (numSamples))
    t_min_numerator = sp.multiply((v1ax_Arr-v2ax_Arr),(x1a_Arr-x2a_Arr))+sp.multiply((v1ay_Arr-v2ay_Arr),(y1a_Arr-y2a_Arr))
    t_min_denominator = sp.power(v1ax_Arr-v2ax_Arr,2) + sp.power(v1ay_Arr-v2ay_Arr,2)
    t_min = -sp.real(sp.divide(t_min_numerator,t_min_denominator))
    # isCollision = sp.ones(numTrials)
    d_min = sp.sqrt(sp.real(sp.power((x1a_Arr-x2a_Arr)+sp.multiply((v1ax_Arr-v2ax_Arr),t_min),2) \
            + sp.power((y1a_Arr-y2a_Arr)+sp.multiply((v1ay_Arr-v2ay_Arr),t_min),2)))
    isCollision = sp.array([d_min[i]<sp.multiply(r,2) and t_min[i]>0 for i in range(len(d_min))])

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
    E_i = CalcConsd.energy(m1_Arr,m2_Arr,v1L,v2L)
    E_f = CalcConsd.energy(m1_Arr,m2_Arr,v1L_f,v2L_f)
    p_x_i = CalcConsd.x_momentum(m1_Arr,m2_Arr,v1L,v2L)
    p_y_i = CalcConsd.y_momentum(m1_Arr,m2_Arr,v1L,v2L)
    p_x_f = CalcConsd.x_momentum(m1_Arr,m2_Arr,v1L_f,v2L_f)
    p_y_f = CalcConsd.y_momentum(m1_Arr,m2_Arr,v1L_f,v2L_f)

    #===========================================
    #
    # Lab Frame Misses Time Series
    #
    #===========================================
    print("Beginning Path Phase: ")
    print(str(time.time() - start_time)+" seconds")

    t = sp.linspace(0, 10, timeSteps)

    # Dimension (2(the # of SpatialDimensions), numTrials, timeSteps)
    Position1L_t = sp.repeat(sp.transpose(InitialPosition1L)[:,:,sp.newaxis], timeSteps, axis=2)\
                   + sp.multiply( sp.repeat(sp.transpose(v1L)[:, :, sp.newaxis], timeSteps, axis=2), sp.transpose(t) )
    Position2L_t = sp.repeat(sp.transpose(InitialPosition2L)[:,:,sp.newaxis], timeSteps, axis=2) \
                   + sp.multiply( sp.repeat(sp.transpose(v2L)[:, :, sp.newaxis], timeSteps, axis=2), sp.transpose(t) )

    Velocity1L_t = sp.repeat(sp.transpose(v1L)[:, :, sp.newaxis], timeSteps, axis=2)
    Velocity2L_t = sp.repeat(sp.transpose(v2L)[:, :, sp.newaxis], timeSteps, axis=2)
    # print(sp.shape(m1_Arr))
    print("Final X1 Array:" + str(sp.shape(Position1L_t)))
    print("Final X2 Array:" + str(sp.shape(Position2L_t)))
    print("Final V1 Array:" + str(sp.shape(Velocity1L_t)))
    print("Final V2 Array:" + str(sp.shape(Velocity2L_t)))

    # placeholders so that output is same format as other file
    dt = sp.empty(2)

    # Save arrays to file
    print("Saving: ")
    print(str(time.time() - start_time)+" seconds")
    print(sp.sum(isCollision))
    if trainOrIntermediateFile==0:
        # Training File
        sp.savez("LabValuesTrain_Misses", v1L, v2L, v1L_f, v2L_f, E_i, E_f, p_x_i, p_x_f, p_y_i, p_y_f,
                 t, Position1L_t, Position2L_t, Velocity1L_t, Velocity2L_t, m1_Arr, m2_Arr, dt, isCollision, radius_Arr)
    elif trainOrIntermediateFile==1:
        # Validation File
        sp.savez("LabValuesIntermediate_Misses", v1L, v2L, v1L_f, v2L_f, E_i, E_f, p_x_i, p_x_f, p_y_i, p_y_f,
                 t, Position1L_t, Position2L_t, Velocity1L_t, Velocity2L_t, m1_Arr, m2_Arr, dt, isCollision, radius_Arr)
print("Time to run: ")
print(str(time.time() - start_time)+" seconds")
