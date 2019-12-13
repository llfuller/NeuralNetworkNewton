import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import CalculateConserved as CalcConsd

# Descends from Net_Single_Kinematic_Timestep.py
# Purpose: Check energy and one dimensional momentum prediction accuracy as
#          a result of position and velocity prediction of a single particle.
#          Analyzes an ensemble of single particles.
#          Below observations (at end of code) apply to all neural networks of this type applied to this data structure.

plotHistory = True
trainOnAcceleration = True
batchSize = 32
numEpochs = 1000
numSamples = 1000

#===============================================================================================================
#   Generating Input and Output for Network
#===============================================================================================================
# Generate positions and velocities. Measured in meters and seconds
# row: time || col: x, v, dt
randomPositions  = sp.multiply(sp.rand(numSamples,1),1)
randomVelocities = sp.multiply(sp.rand(numSamples,1),1)
randomAccels = sp.multiply(sp.rand(numSamples,1),0)
randomDeltaTs    = sp.multiply(sp.rand(numSamples,1),1)
randomPositions_val  = sp.multiply(sp.rand(numSamples,1),1)
randomVelocities_val = sp.multiply(sp.rand(numSamples,1),1)
randomAccels_val = sp.multiply(sp.rand(numSamples,1),0)
randomDeltaTs_val    = sp.multiply(sp.rand(numSamples,1),1)
# Calculate final positions and velocities
positions_f = randomPositions+sp.multiply(randomVelocities,randomDeltaTs)\
              +sp.multiply(sp.multiply(randomAccels,0.5),sp.power(randomDeltaTs,2))
velocities_f = randomVelocities+sp.multiply(randomDeltaTs,randomAccels)
positions_val_f = randomPositions_val+sp.multiply(randomVelocities_val,randomDeltaTs_val)\
                  +sp.multiply(sp.multiply(randomAccels_val,0.5),sp.power(randomDeltaTs_val,2))
velocities_val_f = randomVelocities_val+sp.multiply(randomDeltaTs_val,randomAccels_val)
# Define initial and final state tensors
if(trainOnAcceleration):
    state_input = sp.hstack((randomPositions,randomVelocities,randomAccels, randomDeltaTs))
    # state_output = sp.hstack((positions_f, velocities_f,randomAccels, randomDeltaTs))
    state_output = sp.hstack((positions_f, velocities_f,randomAccels))
    val_input = sp.hstack((randomPositions_val,randomVelocities_val,randomAccels_val, randomDeltaTs_val))
    val_output = sp.hstack((positions_val_f,velocities_val_f,randomAccels_val))
    # val_output = sp.hstack((positions_val_f,velocities_val_f,randomAccels_val,randomDeltaTs_val))
else:
    state_input = sp.hstack((randomPositions,randomVelocities, randomDeltaTs))
    state_output = sp.hstack((positions_f, velocities_f))
    # state_output = sp.hstack((positions_f, velocities_f, randomDeltaTs))
    val_input = sp.hstack((randomPositions_val,randomVelocities_val, randomDeltaTs_val))
    val_output = sp.hstack((positions_val_f,velocities_val_f))
    # val_output = sp.hstack((positions_val_f,velocities_val_f,randomDeltaTs_val))
# Can this figure out the rule x_f = x_i + v*dt? Let's see!

#===============================================================================================================
#   Network
#===============================================================================================================

model = Sequential()
dim = sp.shape(state_input)[1]
model.add(Dense(dim, input_dim = dim, activation='linear'))
model.add(Dense(dim, activation='linear'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(dim, activation='linear'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(dim-1, activation='linear'))

#Compile:
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss= 'mse',
              optimizer = opt,
              metrics = ['mean_absolute_percentage_error'])


history = model.fit(state_input, state_output, batch_size= batchSize, epochs=numEpochs,
                    validation_data = (val_input, val_output),
                    use_multiprocessing = True)

model.summary()
model.save("trainedModel_temp.hd5")

#===============================================================================================================
#   E and P Prediction and Comparison
#===============================================================================================================

#Predict next state of system from current state
input_state = val_input
target_state = val_output
model = load_model('trainedModel_temp.hd5')
model.evaluate(x=input_state, y=target_state)
prediction = model.predict(input_state)
print("Axes: (row: time, column: feature)")
print("Input: ")
print(input_state)
print("Prediction: ")
print(prediction)
print("Actual target state: ")
print(target_state)

# Generate masses to assist in energy and momentum calculation
randomM1 = sp.multiply(sp.rand(numSamples,1),1)
randomM1_val = sp.multiply(sp.rand(numSamples,1),1)
velocities_predicted = sp.array([prediction[:,1]]).transpose()

# first time energy and momentum
E_i_train = CalcConsd.energy_one(randomM1, randomVelocities)
p_i_train = CalcConsd.momentum_one(randomM1, randomVelocities)
E_i_val = CalcConsd.energy_one(randomM1_val, randomVelocities_val)
p_i_val = CalcConsd.momentum_one(randomM1_val, randomVelocities_val)
# second time energy and momentum
E_f_train = CalcConsd.energy_one(randomM1, randomVelocities)
p_f_train = CalcConsd.momentum_one(randomM1, randomVelocities)
E_f_val = CalcConsd.energy_one(randomM1_val, randomVelocities_val)
E_f_val_predicted = CalcConsd.energy_one(randomM1_val, velocities_predicted)
p_f_val = CalcConsd.momentum_one(randomM1_val, randomVelocities_val)
p_f_val_predicted = CalcConsd.momentum_one(randomM1_val, velocities_predicted)

# For plotting energy and momentum ratio of target to predicted states after training is done
horiz_Axis = sp.linspace(1,numSamples+1, numSamples)
E_val_ratio = sp.divide((E_f_val_predicted-E_f_val),E_f_val)
p_val_ratio = sp.divide((p_f_val_predicted-p_f_val),p_f_val)
# Statistics of conserved quantity prediction error
E_val_ratio_avg = sp.mean(E_val_ratio)
p_val_ratio_avg = sp.mean(p_val_ratio)
E_val_ratio_std = sp.std(E_val_ratio)
p_val_ratio_std = sp.std(p_val_ratio)
# print("Energy error: "+str(E_val_ratio_avg)+" +- "+str(E_val_ratio_std))
# print("Momentum error: "+str(p_val_ratio_avg)+" +- "+str(p_val_ratio_std))

#===============================================================================================================
#   Plotting + File Output
#===============================================================================================================

# Plotting conserved quantity error ratios over all validation samples
plt.figure()
plt.scatter(horiz_Axis,E_val_ratio,s=10,color='g')
plt.xlabel("Index")
plt.ylabel("Energy Error Ratio")
plt.title("Energy Error Ratio")
axes = plt.gca()
axes.set_ylim(-1.2, 1)
plt.savefig("NSKTPC-Energy" + str(numSamples) + "SamplesAnd" + str(numEpochs) + "Epochs.png")
plt.figure()
plt.scatter(horiz_Axis,p_val_ratio,s=10,color='b')
plt.xlabel("Index")
plt.ylabel("Momentum Error Ratio")
plt.title("Momentum Error Ratio")
axes = plt.gca()
axes.set_ylim(-1.2, 1)
plt.savefig("NSKTPC-Momentum" + str(numSamples) + "SamplesAnd" + str(numEpochs) + "Epochs.png")
if plotHistory == True:
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    axes = plt.gca()
    axes.set_ylim(0, 0.5)
    plt.savefig("NSKTPC-LossConv" + str(numSamples) + "SamplesAnd" + str(numEpochs) + "Epochs.png")

#===============================================================================================================
#   Observations
#===============================================================================================================
# ReLU makes predictions get stuck with zero values. Because of this, its final convergence is bad for those zeros
# For x, v, and dt given, and few training samples, 10,000 epochs to get to around 1% training MAPE (yikes!)
# For 1 linear layer and few training samples, we get there in above 10,000 epochs (but it's overfitting).
# For 4 linear layers and few training samples, we get there in about 3400 epochs (but it's overfitting):
# Input:
# [[1. 1. 1.]
#  [5. 3. 8.]
#  [6. 9. 2.]]
# Prediction:
# [[ 2.0020845  0.99673    1.0001175]
#  [29.017267   2.9833834  7.9979734]
#  [23.996956   8.984949   1.9976271]]
# Actual target state:
# [[ 2.  1.  1.]
#  [29.  3.  8.]
#  [24.  9.  2.]]
# 7500 epochs for 4 leaky ReLU layers, but it convergences very cleanly (training MAPE=0.0782 after 10,000 epochs)
# The validation MAPE is horrible unless we include a lot of training data.
# If using an outlier (30,000) then the model converges very poorly for a majority of the other elements:
# Input:
# [[1.e+00 1.e+00 1.e+00 3.e+04]
#  [5.e+00 3.e+00 8.e+00 3.e+04]
#  [6.e+00 9.e+00 2.e+00 3.e+04]]
# Prediction:
# [[-4.5717525e+00  1.2607073e+01 -6.6143851e+00  2.9985834e+04]
#  [-4.1779537e+00  1.3829241e+01 -8.6422176e+00  2.9976377e+04]
#  [-2.3708251e+00  1.2334856e+01 -8.4674129e+00  2.9963955e+04]]
# Actual target state:
# [[2.0e+00 1.0e+00 1.0e+00 3.0e+04]
#  [2.9e+01 3.0e+00 8.0e+00 3.0e+04]
#  [2.4e+01 9.0e+00 2.0e+00 3.0e+04]]
# Closer to unity values = smaller error when using 4 linear layers and 1 layer
# 4 linear layers seems to do well with low sample and high sample size
# Wide layers (width: 30) with 0.2 and 0.5 dropout are bad for this. 0.0 dropout does okay but not great.
# Worse case scenario with maximum realistic observed x, v, and a for animals:
#   200, 40, 10, 1 for x, v, a, dt gives validation inaccuracy of above 85% (between 45 and 220% roughly)
# Better scenario: 27% to 33% mean inaccuracy
#    40, 20, 10, 1
# Best case: about 7.5% mean inaccuracy
#    1, 1, 1, 1
# For 0.1, 0.1, 0.1, 0.1: lots of fluctuation between 3% and 10%. If you look at the numbers, it's predicting very well!
# Learned from Net_Single_Timestep and here that permanent zeroes are difficult to deal with. Better to leave out those values if possible.
# Large timesteps bad prediction of velocity and position because the variables are not proportional (maybe fault of disproportionate
#    weighting in cost function)
# If using 0 acceleration in learning and testing, then with two leaky relus and two linear, the v_loss is high (117911.3672)
#    but the median energy error seems to be -0.75 and for momentum it's -0.5:
# Input:
# [[0.23473219 0.4850938  0.         0.9671458 ]
#  [0.56169189 0.81061272 0.         0.56001147]
#  [0.75900103 0.98021628 0.         0.78876037]
#  ...
#  [0.86007883 0.54120019 0.         0.66188895]
#  [0.1415168  0.89622891 0.         0.97582569]
#  [0.38533917 0.42657186 0.         0.26574933]]
# Prediction:
# [[ 8.2257998e-01  2.4459267e-01 -6.6936389e-04  4.2485178e-01]
#  [ 9.4338417e-01  4.3358466e-01 -6.7810528e-04  2.7368405e-01]
#  [ 1.2841562e+00  5.6322879e-01 -8.0493838e-04  3.8835788e-01]
#  ...
#  [ 8.9309716e-01  3.5720924e-01 -4.4855662e-04  3.3239973e-01]
#  [ 1.1221727e+00  4.3109557e-01 -9.7037666e-04  4.2944944e-01]
#  [ 3.9980367e-01  1.9838744e-01 -3.8381852e-04  1.2834665e-01]]
# Actual target state:
# [[0.70388863 0.4850938  0.         0.9671458 ]
#  [1.01564431 0.81061272 0.         0.56001147]
#  [1.53215679 0.98021628 0.         0.78876037]
#  ...
#  [1.21829326 0.54120019 0.         0.66188895]
#  [1.01607999 0.89622891 0.         0.97582569]
#  [0.49870036 0.42657186 0.         0.26574933]]
# Decreasing batch size to 5 made the results worse.