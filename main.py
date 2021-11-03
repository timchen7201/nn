from src.mnist import Mnist
from src.model import Model

epoch = 10

#activation
activation = 'Relu'
output_activation = 'Softmax'

layer_dims = [784, 64, 10]
if __name__ == '__main__':
    X_train,Y_train = Mnist().reshape().get()
    model = Model(layer_dims, activation = activation,output_activation= output_activation)
    model.train(X_train, Y_train)