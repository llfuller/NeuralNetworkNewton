import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, PReLU, SimpleRNN, LSTM
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Observations: Larger, wider networks => Slower real time and epoch convergence to zero inaccuracy
#              Quicker convergence initially
#              Multiplication rule learned easily (2 decimal places for shallow narrow network!
#              Has a very hard time learning from full zero target
#              Easily learns identity weights for target = input, as well as alternating 5 rule
#              Maybe creating a custom cost function would help?

plotHistory = True
batchSize = 32
numEpochs = 10000
learnIdentity = False
learnZeros = False
learnAlternating = False
learnCube = True

model = Sequential()
model.add(Dense(9, input_dim = 9, activation='linear'))
model.add(LeakyReLU(alpha=0.3))
for i in range(50):
    model.add(Dense(3, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
model.add(Dense(9, activation='linear'))
model.add(LeakyReLU(alpha=0.3))



#Compile:
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss= 'mse',
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
state_input_val = sp.multiply(sp.ones((2,9)), 3*stateMultiplier)
state_output = state_input
state_output_val = state_input_val

if not learnIdentity:
    if learnZeros:
        state_output = sp.zeros((2,9))#state_input+0*sp.multiply(alternating5,stateMultiplier)
        state_output_val = sp.zeros((2,9))#3*state_input+0*sp.multiply(alternating5,3*stateMultiplier)
    if learnAlternating:
        state_output = sp.multiply(alternating5,stateMultiplier)
        state_output_val = sp.multiply(alternating5,3*stateMultiplier)
    if learnCube:
        state_input = sp.array([sp.arange(9)]).astype(float)
        state_input_val = sp.array([sp.power(sp.arange(9),3)]).astype(float)
        state_output = sp.array([sp.arange(9,18)]).astype(float)
        state_output_val = sp.array([sp.power(sp.arange(9,18),3)]).astype(float)
print(state_input)
print(sp.shape(state_input))
history = model.fit(state_input, state_output, batch_size= batchSize, epochs=numEpochs,
                    validation_data = (state_input_val, state_output_val),
                    use_multiprocessing = True)

model.summary()
model.save("trainedModel_temp.hd5")

#Predict next state of system from current state
input_state = state_input_val
target_state = state_output_val
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
    axes = plt.gca()
    axes.set_ylim(0, 1)
    plt.savefig("NST-Output/NST_Loss_Convergence.png")
    #plt.show()

outputFile =  open("NST-Output/NSTOutput.txt","w+")

