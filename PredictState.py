from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import scipy as sp
import matplotlib.pyplot as plt

# Predict next state of system from current state
initial_state = sp.array([[[0,0,1,1,
                         1,1,1,1,
                         1,1,1]]])
final_state = sp.array([[[1,1,2,2,
                        1,1,1,1,
                        1,1,1]]])
print(sp.shape(initial_state))
model = load_model('trainedModel_MSL_SquareInp_10000.hd5')
model.evaluate(x=initial_state, y=final_state)
# plot_model(model, to_file='model.png')