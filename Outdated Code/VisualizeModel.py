from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import pydot
import graphviz
import matplotlib.pyplot as plt

# Not working:
model = load_model('trainedModelMAPE_10000.hd5')
plot_model(model, to_file='model.png')



# history = model.history

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()