import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Findings:
# ReLU makes this get some stuck zero values. Because of this, its final convergence is bad for those zeros
# For x, v, and dt given, 10,000 epochs to get to around 1% training MAPE (yikes!)
# For 1 linear layer, we get there in above 10,000 epochs.
# For 4 linear layers, we get there in about 3400 epochs:
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
# The validation MAPE is horrible
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

plotHistory = True
batchSize = 32
numEpochs = 1000

# row: time; col: x, v, dt
numSamples = 500
# meters and seconds
randomPositions  = sp.multiply(sp.rand(numSamples,1),0.1)
randomVelocities = sp.multiply(sp.rand(numSamples,1),0.1)
randomAccels = sp.multiply(sp.rand(numSamples,1),0.1)
randomDeltaTs    = sp.multiply(sp.rand(numSamples,1),0.1)
state_input = sp.hstack((randomPositions,randomVelocities,randomAccels, randomDeltaTs))
state_output = sp.hstack((randomPositions+sp.multiply(randomVelocities,randomDeltaTs)+sp.multiply(sp.multiply(randomAccels,0.5),sp.power(randomDeltaTs,2))
                          ,randomVelocities+sp.multiply(randomDeltaTs,randomAccels),randomAccels, randomDeltaTs))
randomPositions_val  = sp.multiply(sp.rand(numSamples,1),0.1)
randomVelocities_val = sp.multiply(sp.rand(numSamples,1),0.1)
randomAccels_val = sp.multiply(sp.rand(numSamples,1),0.1)
randomDeltaTs_val    = sp.multiply(sp.rand(numSamples,1),0.1)
val_input = sp.hstack((randomPositions_val,randomVelocities_val,randomAccels_val, randomDeltaTs_val))
val_output = sp.hstack((randomPositions_val+sp.multiply(randomVelocities_val,randomDeltaTs_val)+sp.multiply(sp.multiply(randomAccels_val,0.5),sp.power(randomDeltaTs_val,2))
                          ,randomVelocities_val+sp.multiply(randomDeltaTs_val,randomAccels_val),randomAccels_val,randomDeltaTs_val))
# state_input = sp.array([[1.0,1,1],[5,3,8],[6,9,2],[4,7,3],[9,9,9],[17,2,6]])
# state_output = sp.array([[2.0,1,1],[29,3,8],[24,9,2],[25,7,3],[90,9,9],[29,2,6]])
# val_input = sp.array([[5.0,2,7],[14,1,10],[-7,8,13]])
# val_output = sp.array([[19.0,2,7],[24,1,10],[97,8,13]])
# Can this figure out the rule x_f = x_i + v*dt?


dropoutRate = 0.0
model = Sequential()
model.add(Dense(4, input_dim = 4, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(4, activation='linear'))




#Compile:
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss= 'mean_absolute_percentage_error',
              optimizer = opt,
              metrics = ['mean_absolute_percentage_error'])


history = model.fit(state_input, state_output, batch_size= batchSize, epochs=numEpochs,
                    validation_data = (val_input, val_output),
                    use_multiprocessing = True)

model.summary()
model.save("trainedModel_temp.hd5")

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

if plotHistory == True:
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()