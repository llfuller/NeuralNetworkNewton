import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import scipy as sp
#Store all variables ffor all timesteps in one big (numSamples, numTimeSteps) vector.

d_train = sp.load("LabValuesTrain.npz")
d_test = sp.load("LabValuesIntermediate.npz")

numSamples = 100000
Position1L_t=d_train['arr_11'][:,:numSamples,:]
Position2L_t=d_train['arr_12'][:,:numSamples,:]
Velocity1L_t=d_train['arr_13'][:,:numSamples,:]
Velocity2L_t=d_train['arr_14'][:,:numSamples,:]
E_i=d_train['arr_4'][:numSamples]
E_f=d_train['arr_5'][:numSamples]
p_x_i=d_train['arr_6'][:numSamples]
p_x_f=d_train['arr_7'][:numSamples]
p_y_i=d_train['arr_8'][:numSamples]
p_y_f=d_train['arr_9'][:numSamples]

Position1L_t_test=d_test['arr_11'][:,:numSamples,:]
Position2L_t_test=d_test['arr_12'][:,:numSamples,:]
Velocity1L_t_test=d_test['arr_13'][:,:numSamples,:]
Velocity2L_t_test=d_test['arr_14'][:,:numSamples,:]
E_i_test=d_test['arr_4'][:numSamples]
E_f_test=d_test['arr_5'][:numSamples]
p_x_i_test=d_test['arr_6'][:numSamples]
p_x_f_test=d_test['arr_7'][:numSamples]
p_y_i_test=d_test['arr_8'][:numSamples]
p_y_f_test=d_test['arr_9'][:numSamples]

# print(sp.shape(Position1L_t_test))

hugeArray_train = sp.concatenate((Position1L_t[0][:,:-1],Position1L_t[1][:,:-1],
                           Position2L_t[0][:,:-1],Position2L_t[1][:,:-1],
                           Velocity1L_t[0][:,:-1],Velocity1L_t[1][:,:-1],
                           Velocity2L_t[0][:,:-1],Velocity2L_t[1][:,:-1]),
                           axis=1)
hugeArray_train_target = sp.concatenate((Position1L_t[0][:,1:],Position1L_t[1][:,1:],
                           Position2L_t[0][:,1:],Position2L_t[1][:,1:],
                           Velocity1L_t[0][:,1:],Velocity1L_t[1][:,1:],
                           Velocity2L_t[0][:,1:],Velocity2L_t[1][:,1:]),
                           axis=1)
hugeArray_test = sp.concatenate((Position1L_t_test[0][:,:-1],Position1L_t_test[1][:,:-1],
                           Position2L_t_test[0][:,:-1],Position2L_t_test[1][:,:-1],
                           Velocity1L_t_test[0][:,:-1],Velocity1L_t_test[1][:,:-1],
                           Velocity2L_t_test[0][:,:-1],Velocity2L_t_test[1][:,:-1]),
                           axis=1)
hugeArray_test_target = sp.concatenate((Position1L_t_test[0][:,1:],Position1L_t_test[1][:,1:],
                           Position2L_t_test[0][:,1:],Position2L_t_test[1][:,1:],
                           Velocity1L_t_test[0][:,1:],Velocity1L_t_test[1][:,1:],
                           Velocity2L_t_test[0][:,1:],Velocity2L_t_test[1][:,1:]),
                           axis=1)
print(sp.shape(hugeArray_train))
# # Train on 100000 samples, validate on 10000 samples
# # Each number is a (28,28) grid
print("X Training shape: " +str(hugeArray_train.shape)) # (60 000, 28, 28)
print("X Testing shape: " +str(hugeArray_test.shape)) # (10 000, 28, 28)
print("Y Training shape: " + str(hugeArray_train_target.shape)) #(60000,)
print("Y Testing shape: " + str(hugeArray_test_target.shape)) #(10000)

# x_train= x_train/255.0 #normalization
# x_test = x_test/255.0

model = Sequential()
model.add(Flatten(data_format=None))
model.add(Dense(len(hugeArray_train[0]), input_shape=sp.shape(hugeArray_train[0]), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.add(Dense(len(hugeArray_train[0])))
#Compile:
opt = tf.keras.optimizers.SGD(lr=0.00001, momentum = 0.0)
# Note: even though this isn't used: mean_squared_error = mse
model.compile(loss= 'mean_squared_error',
              optimizer = opt,
              metrics = ['accuracy'])
model.fit(hugeArray_train, hugeArray_train_target, epochs=2, validation_data = (hugeArray_test, hugeArray_test_target))

