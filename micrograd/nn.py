import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin= True): # nin = Inputs to a single neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # random list of nin size
        self.b = Value(0)
        self.nonlin= nonlin

    def __call__(self, x):
        act= sum(
            (wi*xi for wi, xi in zip(self.w, x)), # dot product of W.X
            self.b
            )
        return act.relu() if self.nonlin else act   # Returns single scaler 
    
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self): # RELU/Linear Neuron with #x of weights or nin
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"
            
class Layer():

    def __init__(self, nin, nout, **kwargs):
        """
        A list of neurons objects
        where each neuron has nin inputs
        """
        self.neurons= [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        out= [n(x) for n in self.neurons] # list of activation value for each neuron
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        """
        A list of paraemters in each neuron in layer
        """
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts # [<# inputs>] + [4,4,1] -> 4,4,1 is neurons in each layer

        # Layers -> A list
        # Each layers has its inputs and outputs
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
