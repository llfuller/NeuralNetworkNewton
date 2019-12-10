import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import CalculateConserved as CalcConsd
import time
start_time = time.time()

# Predicts whether an initial state leads to a collision or not.
# Author: Lawson Fuller

#===============================================================================================================
#   Run Parameters
#===============================================================================================================

d_train = sp.load("LabValuesTrain_Misses.npz")
d_test = sp.load("LabValuesIntermediate_Misses.npz")

plotHistory = True
numSamples = 10000
T = 199 # max number of timesteps in matrix
batchSize = 32
numEpochs = 100

#===============================================================================================================
#   Import Simulation Data and Preprocess
#===============================================================================================================

Position1L_t= sp.transpose(d_train['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timsteps, features)
Position2L_t= sp.transpose(d_train['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t= sp.transpose(d_train['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t= sp.transpose(d_train['arr_14'][:,:numSamples,:], (1,2,0))
isCollision_Arr  = d_train['arr_18'] [:numSamples]
radius_Arr  = sp.array([d_train['arr_19'] [:numSamples]] ).transpose()
# isCollision_Arr= sp.delete( sp.repeat(d_test['arr_18'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
#                    [t for t in range(1,sp.shape(Velocity1L_t)[1])],1)
# radius_Arr= sp.delete( sp.repeat(d_test['arr_19'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
#                    [t for t in range(1,sp.shape(Velocity1L_t)[1])],1)
Position1L_first= sp.delete(Position1L_t, [t for t in range(1,sp.shape(Position1L_t)[1])],1)
Position2L_first= sp.delete(Position2L_t, [t for t in range(1,sp.shape(Position2L_t)[1])],1)
Velocity1L_first= sp.delete(Velocity1L_t, [t for t in range(1,sp.shape(Velocity1L_t)[1])],1)
Velocity2L_first= sp.delete(Velocity2L_t, [t for t in range(1,sp.shape(Velocity2L_t)[1])],1)
a = sp.array(sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],T,axis=1))
# m1_Arr= sp.delete( sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
#                    [t for t in range(1,sp.shape(Velocity1L_t)[1]-1)],1)
# m2_Arr= sp.delete( sp.repeat(d_test['arr_16'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
#                    [t for t in range(1,sp.shape(Velocity1L_t)[1]-1)],1)
# dt_Arr=sp.repeat(d_train['arr_17'][:numSamples,sp.newaxis],T,axis=1)
# E_i=d_train['arr_4'][:numSamples]
# E_f=d_train['arr_5'][:numSamples]
# p_x_i=d_train['arr_6'][:numSamples]
# p_x_f=d_train['arr_7'][:numSamples]
# p_y_i=d_train['arr_8'][:numSamples]
# p_y_f=d_train['arr_9'][:numSamples]

Position1L_t_val= sp.transpose(d_test['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timesteps, features)
Position2L_t_val= sp.transpose(d_test['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t_val= sp.transpose(d_test['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t_val= sp.transpose(d_test['arr_14'][:,:numSamples,:], (1,2,0))
isCollision_Arr_val  = d_test['arr_18'] [:numSamples]
radius_Arr_val  = sp.array([d_test['arr_19'] [:numSamples]]).transpose()
Position1L_first_val= sp.delete(Position1L_t_val, [t for t in range(1,sp.shape(Position1L_t_val)[1])],1)
Position2L_first_val= sp.delete(Position2L_t_val, [t for t in range(1,sp.shape(Position2L_t_val)[1])],1)
Velocity1L_first_val= sp.delete(Velocity1L_t_val, [t for t in range(1,sp.shape(Velocity1L_t_val)[1])],1)
Velocity2L_first_val= sp.delete(Velocity2L_t_val, [t for t in range(1,sp.shape(Velocity2L_t_val)[1])],1)
# m1_Arr_val= sp.delete( sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
#                        [t for t in range(1,sp.shape(Velocity1L_t_val)[1]-1)],1)
# m2_Arr_val= sp.delete( sp.repeat(d_test['arr_16'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
#                        [t for t in range(1,sp.shape(Velocity1L_t_val)[1]-1)],1)
# dt_Arr_val=sp.repeat(d_test['arr_17'][:numSamples,sp.newaxis],T,axis=1)
E_i_val=d_test['arr_4'][:numSamples]
E_f_val=d_test['arr_5'][:numSamples]
p_x_i_val=d_test['arr_6'][:numSamples]
p_x_f_val=d_test['arr_7'][:numSamples]
p_y_i_val=d_test['arr_8'][:numSamples]
p_y_f_val=d_test['arr_9'][:numSamples]

#===============================================================================================================
#   Collect Input and Output for Network
#===============================================================================================================

# Dimensions of (samples, timesteps, features)
input_Arr = sp.dstack((Position1L_first, Position2L_first, Velocity1L_first, Velocity2L_first,
                       radius_Arr))[:,0,:]
target_Arr = isCollision_Arr
input_Arr_val = sp.dstack((Position1L_first_val, Position2L_first_val, Velocity1L_first_val, Velocity2L_first_val,
                        radius_Arr_val))[:,0,:]
target_Arr_val = isCollision_Arr_val

#===============================================================================================================
#   Network
#===============================================================================================================

# Training existing model. Comment out if you do not wish to do this.
model = load_model("trainedModel_temp.hd5")

model = Sequential()
model.add(Dense(sp.shape(input_Arr)[1]*2, input_shape= sp.shape(input_Arr[0]), activation='linear'))
model.add(Dense(sp.shape(input_Arr)[1]*2, activation='sigmoid'))
# for i in range(50):
#     model.add(Dense(sp.shape(input_Arr)[1], activation='linear'))
#
#     # model.add(LeakyReLU(alpha=0.3))
model.add(Dense(2, activation='sigmoid'))

#Compile:
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)
model.compile(loss= 'mean_absolute_error',
              optimizer = opt,
              metrics = ['accuracy'])
history = model.fit(input_Arr, target_Arr, batch_size= batchSize, epochs=numEpochs,
                    validation_data = (input_Arr_val, target_Arr_val),
                    use_multiprocessing = True)

model.save("NCEE-Output\\trainedModel_NCC.hd5")

print("Time to run: ")
print(str(time.time() - start_time)+" seconds")

classes = sp.array([0,1])
pred_from_val_input = model.predict_classes(input_Arr_val)
falseZero = 0
falseOne = 0
trueZero = 0
trueOne = 0
for i in range(len(target_Arr_val)):
    if target_Arr_val[i] == 1:
        if pred_from_val_input[i] ==1:
            trueOne+=1
        else:
            falseZero += 1
    else:
        if pred_from_val_input[i] ==1:
            falseOne+=1
        else:
            trueZero += 1
print((trueOne, falseOne))
print((falseZero, trueZero))
# con_mat = tf.math.confusion_matrix(labels=classes, predictions=pred_from_val_input, num_classes=2, dtype=int)
# print(con_mat[0])
