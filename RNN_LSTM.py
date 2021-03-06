import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# This code is faulty because it learns using collision point, which does carry properly into the next timestep
#   This code doesn't have an LSTM, which may help it use the collision point properly.

plotHistory = True

d_train = sp.load("LabValuesTrain.npz")
d_test = sp.load("LabValuesIntermediate.npz")

numSamples = 1000
T = 199 # max number of timesteps in matrix
batchSize = 32
numEpochs = 100

isCollision = sp.ones((T, 11))

Position1L_t= sp.transpose(d_train['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timsteps, features)
Position2L_t= sp.transpose(d_train['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t= sp.transpose(d_train['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t= sp.transpose(d_train['arr_14'][:,:numSamples,:], (1,2,0))
m1_Arr=sp.repeat(d_train['arr_15'][:numSamples,sp.newaxis],T,axis=1)
m2_Arr=sp.repeat(d_train['arr_16'][:numSamples,sp.newaxis],T,axis=1)
dt_Arr=sp.repeat(d_train['arr_17'][:numSamples,sp.newaxis],T,axis=1)
E_i=d_train['arr_4'][:numSamples]
E_f=d_train['arr_5'][:numSamples]
p_x_i=d_train['arr_6'][:numSamples]
p_x_f=d_train['arr_7'][:numSamples]
p_y_i=d_train['arr_8'][:numSamples]
p_y_f=d_train['arr_9'][:numSamples]
print(sp.shape(m1_Arr))
print(sp.shape(m2_Arr))

Position1L_t_test= sp.transpose(d_test['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timsteps, features)
Position2L_t_test= sp.transpose(d_test['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t_test= sp.transpose(d_test['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t_test= sp.transpose(d_test['arr_14'][:,:numSamples,:], (1,2,0))
m1_Arr_test=sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],T,axis=1)
m2_Arr_test=sp.repeat(d_test['arr_16'][:numSamples,sp.newaxis],T,axis=1)
dt_Arr_test=sp.repeat(d_test['arr_17'][:numSamples,sp.newaxis],T,axis=1)
E_i_test=d_test['arr_4'][:numSamples]
E_f_test=d_test['arr_5'][:numSamples]
p_x_i_test=d_test['arr_6'][:numSamples]
p_x_f_test=d_test['arr_7'][:numSamples]
p_y_i_test=d_test['arr_8'][:numSamples]
p_y_f_test=d_test['arr_9'][:numSamples]

alternatingZeros=sp.ones((numSamples,T-1,11))
for i in range(0,T-1):
    if i%2==1:
        alternatingZeros[:,i,:]=0
secondHalfZeroes = sp.ones((numSamples,T-1,11))
for i in range(T-1):
    secondHalfZeroes[:,i,:] = (i<(T-1)/2)*9000

print("Position Vector: " +str(sp.shape(Position1L_t)))
hugeArray_train = sp.dstack((Position1L_t[:,:-1], Position2L_t[:,:-1],
                                      Velocity1L_t[:,:-1], Velocity2L_t[:,:-1],dt_Arr[:,:-1]))
hugeArray_train_target = sp.dstack((Position1L_t[:,1:], Position2L_t[:,1:],
                                         Velocity1L_t[:,1:], Velocity2L_t[:,1:],sp.multiply(dt_Arr[:,1:],0)))
hugeArray_test = sp.dstack((Position1L_t_test[:,:-1], Position2L_t_test[:,:-1],
                                 Velocity1L_t_test[:,:-1], Velocity2L_t_test[:,:-1],dt_Arr_test[:,:-1]))
hugeArray_test_target = sp.dstack((Position1L_t_test[:,1:], Position2L_t_test[:,1:],
                                        Velocity1L_t_test[:,1:], Velocity2L_t_test[:,1:],sp.multiply(dt_Arr_test[:,:-1],0)))
print(sp.shape(hugeArray_train))

#==================================================
# # Predict next state of system from current state
# input_state = sp.array([hugeArray_train[0]])
# target_state = sp.array([hugeArray_train_target[0]])
# model = load_model('trainedModel_temp.hd5')
# model.evaluate(x=input_state, y=target_state)
# prediction = model.predict(input_state)
# print("Axes: (row: time, column: feature)")
# print("Input: ")
# print(input_state)
# print("Prediction: ")
# print(prediction)
# print("Actual target state: ")
# print(target_state)

#=================================================

print("Training input shape: " +str(hugeArray_train.shape)) # (numSamples, T-1, 11)
print("Testing input shape: " +str(hugeArray_test.shape)) # (numSamples, T-1, 11)
print("Training Target shape: " + str(hugeArray_train_target.shape)) #(numSamples, T-1, 11)
print("Testing Target shape: " + str(hugeArray_test_target.shape)) #(numSamples, T-1, 11)

# model = Model(inputs= hugeArray_train, outputs = hugeArray_train_target)
model = Sequential()
# training matrix (hugeArray) has dimension (numSamples, T-1, 11)
# batchSize = numSamples = 1000 <--- This is where the 1000 comes from
# T-1 = 198  <--- Source of 198
# 11 <--- number of features
# model.add(LSTM(sp.shape(hugeArray_train)[2], input_shape= sp.shape(hugeArray_train[0]) ) )
model.add(Dense(sp.shape(hugeArray_train)[2], input_shape= sp.shape(hugeArray_train[0]), activation='linear'))
# model.add(LeakyReLU(alpha=0.3))
# model.add(LeakyReLU(input_shape = sp.shape(hugeArray_train[0]),alpha=1))
print(hugeArray_train[0])
model.add(LeakyReLU(alpha=0.3))
# model.add(Dense(sp.shape(hugeArray_train)[2], input_shape = sp.shape(hugeArray_train[0])))
# model.add(Dense(sp.shape(hugeArray_train)[2]))
#
# model.add(LeakyReLU(alpha=1))


# test matrix has dimension (numSamples, T-1, 11)





# model.add(Flatten(data_format=None))
# model.add(SimpleRNN(12, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))


# for i in range(100):
#     #model.add(LeakyReLU(alpha=0.3))
#     model.add(Dense(len(hugeArray_train[0]), activation='relu'))

#Compile:
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss= 'mean_absolute_percentage_error',
              optimizer = opt,
              metrics = ['mean_absolute_percentage_error'])


# loss: 0.1379 - mean_absolute_percentage_error: 99.3193 - val_loss: 0.1622 - val_mean_absolute_percentage_error: 98.5299
history = model.fit(hugeArray_train, sp.zeros((numSamples,T-1,9)), batch_size= batchSize, epochs=numEpochs,
                    validation_data = (hugeArray_test, sp.zeros((numSamples,T-1,9))),
                    use_multiprocessing = True)
# lReLU_weights1 = model.layers[0].get_weights()[0]
# lReLU_weights2 = model.layers[1].get_weights()[0]
# print(lReLU_weights1)
# print(lReLU_weights2)

model.save("trainedModel_temp.hd5")

if plotHistory == True:
    plt.figure()
    # # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
model.summary()