from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import scipy as sp
import matplotlib.pyplot as plt

d_test = sp.load("LabValuesIntermediate.npz")
model = load_model('NBSTE-Output\\trainedModel_NCEE.hd5')

numSamples = 1 # Drawing plot for only one particle
Position1L_t_val= sp.transpose(d_test['arr_11'][:,:numSamples,:], (1,2,0)) # (samples, timesteps, features)
Position2L_t_val= sp.transpose(d_test['arr_12'][:,:numSamples,:], (1,2,0))
Velocity1L_t_val= sp.transpose(d_test['arr_13'][:,:numSamples,:], (1,2,0))
Velocity2L_t_val= sp.transpose(d_test['arr_14'][:,:numSamples,:], (1,2,0))
Position1L_firstSecond_val= sp.delete(Position1L_t_val, [t for t in range(2,sp.shape(Velocity1L_t_val)[1])],1)
Position2L_firstSecond_val= sp.delete(Position2L_t_val, [t for t in range(2,sp.shape(Velocity1L_t_val)[1])],1)
Velocity1L_firstSecond_val= sp.delete(Velocity1L_t_val, [t for t in range(2,sp.shape(Velocity1L_t_val)[1])],1)
Velocity2L_firstSecond_val= sp.delete(Velocity2L_t_val, [t for t in range(2,sp.shape(Velocity1L_t_val)[1])],1)

dt_Arr_val=sp.delete( sp.repeat(d_test['arr_17'][:numSamples,sp.newaxis],sp.shape(Velocity1L_t_val)[1],axis=1),
                   [t for t in range(2,sp.shape(Velocity1L_t_val)[1])],1)

initial_state = sp.dstack((Position1L_firstSecond_val, Velocity1L_firstSecond_val))[:,0,:]

plotThisX = [initial_state[0,0]]
plotThisY = [initial_state[0,1]]
statesCovered = [initial_state]
for i in range(100):
    predictFromThis = sp.array([sp.append(statesCovered[i], dt_Arr_val[0,0]) for j in range(1)])
    prediction = model.predict(predictFromThis)
    tempXArray = prediction[0,0]
    tempYArray = prediction[0,1]
    plotThisX.append(tempXArray)
    plotThisY.append(tempYArray)
    statesCovered.append(prediction)

plt.figure()
plt.plot(plotThisX, plotThisY)
plt.show()
print("hi")

# plot_model(model, to_file='model.png')