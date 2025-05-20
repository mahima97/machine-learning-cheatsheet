import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize random seed for reproducibility
nnfs.init()

# Load dataset
X, y = spiral_data(100, 3)

# Print unique values in y to inspect the label range
print("Unique values in y:", np.unique(y))

# Layer Definitions
class Layer_Dense:
    """Dense Layer for neural network"""
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)  # Derivative w.r.t weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)  # Derivative w.r.t biases
        self.dinput = np.dot(dvalues, self.weights.T)  # Derivative w.r.t input


# Activation Functions
class Activation_ReLU:
    """ReLU Activation Function"""
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinput = dvalues.copy()
        self.dinput[self.output <= 0] = 0  # Gradient for ReLU


class Activation_Softmax:
    """Softmax Activation Function"""
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinput = dvalues.copy()


# Loss Function Class
class Loss:
    """Base Loss Class"""
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    """Categorical Crossentropy Loss"""
    def forward(self, y_pred, y_gt):
        samples = len(y_pred)

        # Ensure labels are within valid range for one-hot encoding
        if np.max(y_gt) >= y_pred.shape[1] or np.min(y_gt) < 0:
            raise ValueError(f"Labels must be between 0 and {y_pred.shape[1] - 1}, but got labels outside this range.")

        # One-hot encode the labels if they are integers
        if len(y_gt.shape) == 1:
            y_gt = np.eye(y_pred.shape[1])[y_gt]

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # To avoid log(0)
        correct_confidence = np.sum(y_pred_clipped * y_gt, axis=1)  # Element-wise multiplication

        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

    def backward(self, y_pred, y_gt):
        samples = len(y_pred)

        # Ensure labels are within valid range for one-hot encoding
        if np.max(y_gt) >= y_pred.shape[1] or np.min(y_gt) < 0:
            raise ValueError(f"Labels must be between 0 and {y_pred.shape[1] - 1}, but got labels outside this range.")

        # One-hot encode the labels if they are integers
        if len(y_gt.shape) == 1:
            y_gt = np.eye(y_pred.shape[1])[y_gt]

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinput = (y_pred_clipped - y_gt) / samples  # Derivative of loss
        return self.dinput


# Optimizer Class (Adam)
class Optimizer_Adam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, layer):
        if self.m is None:
            self.m = [np.zeros_like(w) for w in layer.weights]
            self.v = [np.zeros_like(w) for w in layer.weights]

        self.t += 1
        for i in range(len(layer.weights)):
            # Update moving averages
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.dweights
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.dweights ** 2)

            # Correct bias
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update weights and biases
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            layer.biases -= self.learning_rate * layer.dbiases


# Training Function
def train_neural_network(X, y, epochs=10, learning_rate=0.001):
    # Initialize Layers, Activations, Loss Function, Optimizer
    dense1 = Layer_Dense(2, 9)
    dense2 = Layer_Dense(9, 15)
    dense3 = Layer_Dense(15, 2)
    activation_relu = Activation_ReLU()
    output_layer = Activation_Softmax()

    loss_function = Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        # Forward Pass
        dense1.forward(X)
        activation_relu.forward(dense1.output)
        dense2.forward(activation_relu.output)
        activation_relu.forward(dense2.output)
        dense3.forward(activation_relu.output)
        output_layer.forward(dense3.output)

        # Loss Calculation
        loss = loss_function.calculate(output_layer.output, y)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")

        # Backpropagation
        dvalues = loss_function.backward(output_layer.output, y)
        output_layer.backward(dvalues)
        dense3.backward(output_layer.dinput)
        activation_relu.backward(dense3.dinput)
        dense2.backward(activation_relu.dinput)
        activation_relu.backward(dense2.dinput)
        dense1.backward(activation_relu.dinput)

        # Update weights using Adam optimizer
        optimizer.update(dense1)
        optimizer.update(dense2)
        optimizer.update(dense3)


# Run Training
train_neural_network(X, y, epochs=10, learning_rate=0.001)




# import numpy as np
# import nnfs
# from nnfs.datasets import spiral_data

# nnfs.init()

# X,y = spiral_data(100,3)


# class Layer_Dense:
# 	"""Layer_Dense Initilizes with following arguments
# 		n_input: input dimansion
# 		n_neurons: number of neurons

# 	forward : Function takes in inputs and process the 
# 	"""
# 	def __init__(self, n_inputs, n_neurons):
# 		super(Layer_Dense, self).__init__()
# 		self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
# 		self.biases = np.zeros((1,n_neurons))
# 	def forward(self, inputs):
# 		self.output = np.dot(inputs, self.weights) + self.biases

# class Activation_ReLU:
# 	"""docstring for Activation_ReLU"""
# 	def forward(self, inputs):
# 		self.output = np.maximum(0, inputs)

# class Activation_Softmax:
# 	"""docstring for Activation_Softmax"""
# 	def forward(self, inputs):
# 		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
# 		self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)

# 	def 

# class Loss(object):
# 	"""docstring for Loss"""
# 	def calcualte(self, output, y):
# 		sample_losses = self.forward(output. y)
# 		data_loss = np.mean(sample_losses)
# 		return data_loss


# class Loss_CategoricalCrossentropy(object):
# 	"""docstring for Loss_CategoricalCrossentropy"""
# 	def forward(self, y_pred, y_gt):
# 		samples = len(y_pred)
# 		y_pred_clipped = np.clip(y_pred,1e-7, 1-1e-7)

# 		if len(y_pred.shape)==1:
# 			correct_confidence = y_pred_clipped[range(samples), y_gt]

# 		elif len(y_pred.shape)==2:
# 			correct_confidence = np.sum(y_pred_clipped*y_gt, axis=1)

# 		negative_log_likelihood = -np.log(correct_confidence)
# 		return negative_log_likelihood


# dense1 = Layer_Dense(2,9)
# dense2 = Layer_Dense(9,15)
# dense3 = Layer_Dense(15,2)
# activation = Activation_ReLU()
# output_layer = Activation_Softmax()

# dense1.forward(X)
# l1 = activation.forward(dense1.output)
# dense2.forward(activation.output)
# activation.forward(dense2.output)





		
		
