from src.nn import NeuralNetwork
from src.utils import shuffle,get_batch
from src.optimizer import Optimizer
from tqdm import trange
from src.activation import Relu,Sigmoid,Softmax
import time
class Model:
    def __init__(self,layer_dims,activation='Relu',output_activation='softmax'):
        self._act(activation,output_activation)
        self.NN = NeuralNetwork(
            input_layer_dim = layer_dims[0], 
            hidden_layer_dim = layer_dims[1], 
            output_layer_dim= layer_dims[2],
            activation = self.act,
            output_activation = self.output_act
        )
    def _act(self,activation,output_activation):
        if activation == "Relu":
            self.act= Relu
        elif activation == "Sigmoid":
            self.act= Sigmoid
        else:
            raise ValueError("We only support relu and sigmoid as an activation function")
        if output_activation == "Softmax":
            self.output_act= Softmax

    def train(self, X_train, Y_train, batch_size=32, epochs=1):
        total_data_num = X_train.shape[0]
        batches = total_data_num//batch_size

        for epoch in range(1,epochs+1):
            print(f"Epoch {epoch}")
            X_shuffle, Y_shuffle = shuffle(X_train,Y_train)
            start_time = time.time()
            for current_batch in trange(batches):
                X = get_batch(X_shuffle, current_batch,batch_size,total_data_num)
                out = self.NN.forward_pass(X)

                self.NN.backward_pass(Y_shuffle)

                optimizer = Optimizer()

                optimizer.update(self.NN)