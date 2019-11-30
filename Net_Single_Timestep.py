import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Observations: Larger, wider networks => Slower real time and epoch convergence to zero inaccuracy
#              Quicker convergence initially
#              Multiplication rule learned easily (2 decimal places for shallow narrow network!

plotHistory = True
batchSize = 32
numEpochs = 2000

model = Sequential()
model.add(Dense(9, input_dim = 9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(9, activation='linear'))

#Compile:
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss= 'mean_absolute_percentage_error',
              optimizer = opt,
              metrics = ['mean_absolute_percentage_error'])

alternating5 = sp.ones((2,9))
for i in range(len(alternating5[0])):
    if i%2 == 0:
        alternating5[:,i] = 5

stateMultiplier = sp.ones((2,9))
for i in range(len(alternating5)):
    stateMultiplier[i,:] = i+1

state_input = sp.multiply(sp.ones((2,9)), stateMultiplier)
state_output = sp.multiply(alternating5,stateMultiplier)

history = model.fit(state_input, state_output, batch_size= batchSize, epochs=numEpochs,
                    validation_data = (state_input, state_output),
                    use_multiprocessing = True)

model.summary()
model.save("trainedModel_temp.hd5")

#Predict next state of system from current state
input_state = state_input
target_state = state_output
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

