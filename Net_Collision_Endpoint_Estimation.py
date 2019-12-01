import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import CalculateConserved as CalcConsd
import time
start_time = time.time()

# Predicts final velocities and masses, and therefore also energy and x and y momenta.
# Compares predicted conserved quantities with actual values, then saves plots and records median values.
# Author: Lawson Fuller

#===============================================================================================================
#   Run Parameters
#===============================================================================================================

d_train = sp.load("LabValuesTrain.npz")
d_test = sp.load("LabValuesIntermediate.npz")

plotHistory = True
numSamples = 10000
T = 199 # max number of timesteps in matrix
batchSize = 32
numEpochs = 1000 # Converges around 300 for LeakyReLU + 3 Dense

#===============================================================================================================
#   Import Simulation Data and Preprocess
#===============================================================================================================

# Position1L_t= sp.transpose(d_train['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timsteps, features)
# Position2L_t= sp.transpose(d_train['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t= sp.transpose(d_train['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t= sp.transpose(d_train['arr_14'][:,:numSamples,:], (1,2,0))
# Position1L_firstLast= sp.delete(Position1L_t, [t for t in range(1,sp.shape(Position1L_t)[1]-1)],1)
# Position2L_firstLast= sp.delete(Position2L_t, [t for t in range(1,sp.shape(Position2L_t)[1]-1)],1)
Velocity1L_firstLast= sp.delete(Velocity1L_t, [t for t in range(1,sp.shape(Velocity1L_t)[1]-1)],1)
Velocity2L_firstLast= sp.delete(Velocity2L_t, [t for t in range(1,sp.shape(Velocity2L_t)[1]-1)],1)
a = sp.array(sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],T,axis=1))
m1_Arr= sp.delete( sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
                   [t for t in range(1,sp.shape(Velocity1L_t)[1]-1)],1)
m2_Arr= sp.delete( sp.repeat(d_test['arr_16'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
                   [t for t in range(1,sp.shape(Velocity1L_t)[1]-1)],1)
# dt_Arr=sp.repeat(d_train['arr_17'][:numSamples,sp.newaxis],T,axis=1)
# E_i=d_train['arr_4'][:numSamples]
# E_f=d_train['arr_5'][:numSamples]
# p_x_i=d_train['arr_6'][:numSamples]
# p_x_f=d_train['arr_7'][:numSamples]
# p_y_i=d_train['arr_8'][:numSamples]
# p_y_f=d_train['arr_9'][:numSamples]

# Position1L_t_val= sp.transpose(d_test['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timesteps, features)
# Position2L_t_val= sp.transpose(d_test['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t_val= sp.transpose(d_test['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t_val= sp.transpose(d_test['arr_14'][:,:numSamples,:], (1,2,0))
# Position1L_firstLast_val= sp.delete(Position1L_t_val, [t for t in range(1,sp.shape(Position1L_t_val)[1]-1)],1)
# Position2L_firstLast_val= sp.delete(Position2L_t_val, [t for t in range(1,sp.shape(Position2L_t_val)[1]-1)],1)
Velocity1L_firstLast_val= sp.delete(Velocity1L_t_val, [t for t in range(1,sp.shape(Velocity1L_t_val)[1]-1)],1)
Velocity2L_firstLast_val= sp.delete(Velocity2L_t_val, [t for t in range(1,sp.shape(Velocity2L_t_val)[1]-1)],1)
m1_Arr_val= sp.delete( sp.repeat(d_test['arr_15'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
                       [t for t in range(1,sp.shape(Velocity1L_t_val)[1]-1)],1)
m2_Arr_val= sp.delete( sp.repeat(d_test['arr_16'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t)[1],axis=1),
                       [t for t in range(1,sp.shape(Velocity1L_t_val)[1]-1)],1)
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
input_Arr = sp.dstack((Velocity1L_firstLast, Velocity2L_firstLast, m1_Arr, m2_Arr))[:,0,:]
target_Arr = sp.dstack((Velocity1L_firstLast, Velocity2L_firstLast, m1_Arr, m2_Arr))[:,1,:]
input_Arr_val = sp.dstack((Velocity1L_firstLast_val, Velocity2L_firstLast_val, m1_Arr_val, m2_Arr_val))[:,0,:]
target_Arr_val = sp.dstack((Velocity1L_firstLast_val, Velocity2L_firstLast_val, m1_Arr_val, m2_Arr_val))[:,1,:]

#===============================================================================================================
#   Network
#===============================================================================================================

model = Sequential()
model.add(Dense(sp.shape(input_Arr)[1], input_shape= sp.shape(input_Arr[0]), activation='linear'))
model.add(LeakyReLU(alpha=0.3))
for i in range(5):
    model.add(Dense(sp.shape(input_Arr)[1]*8, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
model.add(Dense(sp.shape(input_Arr)[1], activation='linear'))
model.add(LeakyReLU(alpha=0.3))

#Compile:
opt = tf.keras.optimizers.Adam(lr=1.5e-3, decay=1e-6)
model.compile(loss= 'mean_absolute_percentage_error',
              optimizer = opt,
              metrics = ['mean_absolute_percentage_error'])
history = model.fit(input_Arr, target_Arr, batch_size= batchSize, epochs=numEpochs,
                    validation_data = (input_Arr_val, target_Arr_val),
                    use_multiprocessing = True)

model.save("NCEE-Output\\trainedModel_NCEE.hd5")

#===============================================================================================================
#   E and P Prediction and Comparison
#===============================================================================================================
# model = load_model("trainedModel_temp.hd5")
model.evaluate(x=input_Arr_val, y=target_Arr_val)
prediction = model.predict(input_Arr_val)
predicted_v1 = prediction[:,0:2]
predicted_v2 = prediction[:,2:4]
predicted_m1 = prediction[:,4]
predicted_m2 = prediction[:,5]

# first time energy and momentum
E_val = CalcConsd.energy(m1_Arr_val[:,0], m2_Arr_val[:,0], Velocity1L_firstLast_val[:,1,:],
                         Velocity2L_firstLast_val[:,1,:]) # (sample,timestep (0 or 1),feature)
px_val = CalcConsd.x_momentum(m1_Arr_val[:,0], m2_Arr_val[:,0], Velocity1L_firstLast_val[:,1,:],
                              Velocity2L_firstLast_val[:,1,:])
py_val = CalcConsd.y_momentum(m1_Arr_val[:,0], m2_Arr_val[:,0], Velocity1L_firstLast_val[:,1,:],
                              Velocity2L_firstLast_val[:,1,:])
# second time energy and momentum
E_pred = CalcConsd.energy(predicted_m1, predicted_m2, predicted_v1, predicted_v2)
px_pred = CalcConsd.x_momentum(predicted_m1, predicted_m2, predicted_v1, predicted_v2)
py_pred = CalcConsd.y_momentum(predicted_m1, predicted_m2, predicted_v1, predicted_v2)


# For plotting energy and momentum ratio of target to predicted states after training is done
E_val_ratio = sp.divide((E_pred-E_val),E_val)
px_val_ratio = sp.divide((px_pred-px_val),px_val)
py_val_ratio = sp.divide((py_pred-py_val),py_val)
# Statistics of conserved quantity prediction error (median used because a couple large outliers skew mean)
E_val_ratio_median = sp.median(E_val_ratio)
px_val_ratio_median = sp.median(px_val_ratio)
py_val_ratio_median = sp.median(py_val_ratio)
print("Energy median: "+str(E_val_ratio_median))
print("X Momentum median: "+str(px_val_ratio_median))
print("Y Momentum median: "+str(py_val_ratio_median))
print("A value of negative 1 means the predicted value is a tiny fraction of the real amount")
print("E")
print(E_val[0:3])
print(E_pred[0:3])
print("px")
print(px_val[0:3])
print(px_pred[0:3])
print("py")
print(py_val[0:3])
print(py_pred[0:3])


#===============================================================================================================
#   Plotting
#===============================================================================================================
horiz_Axis = sp.linspace(1,numSamples+1, numSamples)

# Plotting conserved quantity error ratios over all validation samples
plt.figure()
plt.scatter(horiz_Axis,E_val_ratio,s=2,color='g')
plt.xlabel("Index")
plt.ylabel("Energy Error Ratio")
plt.title("Energy Error Ratio")
axes = plt.gca()
axes.set_ylim(-1.2, 1)
plt.savefig("NCEE-Output\\NCEE-EWith"+str(numSamples)+"SamplesAnd"+str(numEpochs)+"Epochs.png")
plt.figure()
plt.scatter(horiz_Axis,px_val_ratio,s=2,color='b')
plt.xlabel("Index")
plt.ylabel("X Momentum Error Ratio")
plt.title("X Momentum Error Ratio")
axes = plt.gca()
axes.set_ylim(-1.2, 1)
plt.savefig("NCEE-Output\\NCEE-p_xWith"+str(numSamples)+"SamplesAnd"+str(numEpochs)+"Epochs.png")
plt.figure()
plt.scatter(horiz_Axis,py_val_ratio,s=2,color='c')
plt.xlabel("Index")
plt.ylabel("Y Momentum Error Ratio")
plt.title("Y Momentum Error Ratio")
axes = plt.gca()
axes.set_ylim(-1.2, 1)
plt.savefig("NCEE-Output\\NCEE-p_yWith"+str(numSamples)+"SamplesAnd"+str(numEpochs)+"Epochs.png")

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
    axes = plt.gca()
    axes.set_ylim(0, 100)
    plt.savefig("NCEE-Output\\NCEE-LossConv" + str(numSamples) + "SamplesAnd" + str(numEpochs) + "Epochs.png")
model.summary()
# plt.show()

#===============================================================================================================
#   File Output
#===============================================================================================================

outputFile =  open("NCEE-Output/NCEEOutput.txt","w+")
outputFile.write("Samples:"+str(numSamples)+"; Epochs:"+str(numEpochs)+"\n")
outputFile.write("Runtime: "+str(time.time() - start_time)+" seconds"+"\n")
outputFile.write("Energy median: "+str(E_val_ratio_median)
                 +"\nX Momentum median: "+str(px_val_ratio_median)
                 +"\nY Momentum median: "+str(py_val_ratio_median)+"\n")
outputFile.write("First array expected, second array predicted:\n")
outputFile.write("E"+str(E_val[0:3])+";"+str(E_pred[0:3])+";"
                 +"\npx"+str(px_val[0:3])+";"+str(px_pred[0:3])+";"
                 +"\npy"+str(py_val[0:3])+";"+str(py_pred[0:3])+";")
outputFile.close()

print("Time to run: ")
print(str(time.time() - start_time)+" seconds")
