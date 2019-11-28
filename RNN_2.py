import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
import scipy as sp

d_train = sp.load("LabValuesTrain.npz")
d_test = sp.load("LabValuesIntermediate.npz")

numSamples = 10000
T = 199 # number of timesteps in matrix

Position1L_t=d_train['arr_11'][:,:numSamples,:].reshape((2,numSamples*T))
Position2L_t=d_train['arr_12'][:,:numSamples,:].reshape((2,numSamples*T))
Velocity1L_t=d_train['arr_13'][:,:numSamples,:].reshape((2,numSamples*T))
Velocity2L_t=d_train['arr_14'][:,:numSamples,:].reshape((2,numSamples*T))
m1_Arr=sp.repeat(d_train['arr_15'][:numSamples],T)
m2_Arr=sp.repeat(d_train['arr_16'][:numSamples],T)
E_i=d_train['arr_4'][:numSamples]
E_f=d_train['arr_5'][:numSamples]
p_x_i=d_train['arr_6'][:numSamples]
p_x_f=d_train['arr_7'][:numSamples]
p_y_i=d_train['arr_8'][:numSamples]
p_y_f=d_train['arr_9'][:numSamples]

Position1L_t_test=d_test['arr_11'][:,:numSamples,:].reshape((2,numSamples*T))
Position2L_t_test=d_test['arr_12'][:,:numSamples,:].reshape((2,numSamples*T))
Velocity1L_t_test=d_test['arr_13'][:,:numSamples,:].reshape((2,numSamples*T))
Velocity2L_t_test=d_test['arr_14'][:,:numSamples,:].reshape((2,numSamples*T))
m1_Arr_test=sp.repeat(d_train['arr_15'][:numSamples],T)
m2_Arr_test=sp.repeat(d_train['arr_16'][:numSamples],T)
E_i_test=d_test['arr_4'][:numSamples]
E_f_test=d_test['arr_5'][:numSamples]
p_x_i_test=d_test['arr_6'][:numSamples]
p_x_f_test=d_test['arr_7'][:numSamples]
p_y_i_test=d_test['arr_8'][:numSamples]
p_y_f_test=d_test['arr_9'][:numSamples]

print("Position Vector: " +str(sp.shape(Position1L_t)))
hugeArray_train = sp.vstack((Position1L_t[:,:-1], Position2L_t[:,:-1],
                                  Velocity1L_t[:,:-1], Velocity2L_t[:,:-1],
                                 m1_Arr[:-1],m2_Arr[:-1])).transpose() # axis 0 are timesteps (new sample very 199 timesteps), axis 1 is value
hugeArray_train_target = sp.vstack((Position1L_t[:,1:], Position2L_t[:,1:],
                                         Velocity1L_t[:,1:], Velocity2L_t[:,1:],
                                         m1_Arr[1:],m2_Arr[1:])).transpose() # axis 0 are timesteps (new sample very 199 timesteps), axis 1 is value
hugeArray_test = sp.vstack((Position1L_t_test[:,:-1], Position2L_t_test[:,:-1],
                                 Velocity1L_t_test[:,:-1], Velocity2L_t_test[:,:-1],
                                 m1_Arr_test[:-1],m2_Arr_test[:-1])).transpose()
hugeArray_test_target = sp.vstack((Position1L_t_test[:,1:], Position2L_t_test[:,1:],
                                        Velocity1L_t_test[:,1:], Velocity2L_t_test[:,1:],
                                        m1_Arr_test[1:],m2_Arr_test[1:])).transpose()
print(sp.shape(hugeArray_train))

print("Training input shape: " +str(hugeArray_train.shape)) # (80000, 198)
print("Testing input shape: " +str(hugeArray_test.shape)) # (80000, 198)
print("Training Target shape: " + str(hugeArray_train_target.shape)) #(80000, 198)
print("Testing Target shape: " + str(hugeArray_test_target.shape)) #(80000, 198)


model = Sequential()
model.add(Flatten(data_format=None))
model.add(Dense(len(hugeArray_train[0]), input_shape=sp.shape(hugeArray_train[0]), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(LeakyReLU(alpha=0.3))
#Compile:
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-3)

model.compile(loss= 'mean_squared_error',
              optimizer = opt,
              metrics = ['accuracy'])
model.fit(hugeArray_train, hugeArray_train_target, epochs=5, validation_data = (hugeArray_test, hugeArray_test_target))

