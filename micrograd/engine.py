class Value:
    """ 
    Applicable for only single scale Value, not tensor, and its gradient 
    Credits: Karpathy's implementation of Micrograd @ https://github.com/karpathy/micrograd
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0       # All scalers have 0 gradient by default
        
        # Gradient function - None by default. Actaul functions update at operations
        self._backward = lambda: None

        self._prev = set(_children) #Tracks the child nodes
        self._op = _op # for visualization purpose

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Handles if somone adds Value + 1 integer
        out = Value(data = self.data+ other.data, _children= (self, other), _op= '+')

        # Function to compute _backward gradients for +
        def _backward():
            # Gradient of sum is sum of gradients - simply
            # Below we simply decompose the gradient of out to its components
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward= _backward # Updating _backward function for Value

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Handles if somone adds Value + 1 integer
        out = Value(data = self.data * other.data, _children = (self, other), _op= '*')

        # Function to compute _backward gradients for *
        def _backward():
            # Apply chain rule
            # Ex for self.grad= Local derivative = Other.data & out.grad = global derivative
            self.grad += other.data * out.grad
            other.grad += self.data* out.grad
        
        out._backward= _backward # Updating _backward function for Value
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(data= self.data**other, _children= (self,), _op= f'**{other}')

        def _backward():
            # Apply power rule on self 
            self.grad += (other * self.data**(other-1)) * out.grad
        
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(
            data= 0 if self.data < 0 else self.data, 
            _children= (self,), 
            _op= 'ReLU'
            )
        
        def _backward(): 
            self.grad += (out.data >0 ) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
