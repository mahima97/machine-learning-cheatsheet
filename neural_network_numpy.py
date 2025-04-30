import numpy as np

class NeuralNetwork:
    def __init__(self, inputSize, layers, outputSize, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32):
        self.layers = layers
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.weights, self.biases = self.initialization()
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Time step

    def initialization(self):
        weights = []
        biases = []
        prev_size = self.inputSize
        for layer_size in self.layers:
            weights.append(np.random.randn(prev_size, layer_size) * 0.01)
            biases.append(np.zeros((1, layer_size)))
            prev_size = layer_size
        weights.append(np.random.randn(prev_size, self.outputSize) * 0.01)
        biases.append(np.zeros((1, self.outputSize)))
        return weights, biases

    def activation_function(self, x, func="relu"):
        if func == "relu":
            return np.maximum(0, x)
        elif func == "relu_derive":
            return (x > 0).astype(float)
        elif func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif func == "sigmoid_derive":
            sig = self.activation_function(x, "sigmoid")
            return sig * (1 - sig)
        elif func == "softmax":
            x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return x / np.sum(x, axis=1, keepdims=True)

    def feedforward(self, X):
        self.zs = []
        self.activations = [X]
        input_data = X
        for i in range(len(self.weights)):
            z = np.dot(input_data, self.weights[i]) + self.biases[i]
            self.zs.append(z)
            if i == len(self.weights) - 1:
                a = self.activation_function(z, "softmax")
            else:
                a = self.activation_function(z, "relu")
            self.activations.append(a)
            input_data = a

    def loss_function(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-15)
        return np.sum(log_likelihood) / m

    def backpropagation(self, X, y):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # output layer error
        delta = self.activations[-1] - y
        grads_w[-1] = np.dot(self.activations[-2].T, delta)
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # backward through hidden layers
        for i in reversed(range(len(self.layers))):
            z = self.zs[i]
            da = self.activation_function(z, "relu_derive")
            delta = np.dot(delta, self.weights[i + 1].T) * da
            grads_w[i] = np.dot(self.activations[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

        return {"dW": grads_w, "dB": grads_b}

    def optimization_function(self, weights, biases, grads, m_w, v_w, m_b, v_b):
        self.t += 1
        new_weights, new_biases = [], []

        for i in range(len(weights)):
            # Update moment estimates
            m_w[i] = self.beta1 * m_w[i] + (1 - self.beta1) * grads['dW'][i]
            v_w[i] = self.beta2 * v_w[i] + (1 - self.beta2) * (grads['dW'][i] ** 2)

            m_b[i] = self.beta1 * m_b[i] + (1 - self.beta1) * grads['dB'][i]
            v_b[i] = self.beta2 * v_b[i] + (1 - self.beta2) * (grads['dB'][i] ** 2)

            # Bias correction
            m_w_hat = m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = v_w[i] / (1 - self.beta2 ** self.t)
            m_b_hat = m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = v_b[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            w_update = self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            b_update = self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            weights[i] -= w_update
            biases[i] -= b_update

        return weights, biases, m_w, v_w, m_b, v_b

    def gradient_decent(self, grads):
        self.weights, self.biases, self.m_w, self.v_w, self.m_b, self.v_b = self.optimization_function(
            self.weights, self.biases, grads, self.m_w, self.v_w, self.m_b, self.v_b
        )

    def train(self, X, y, epoch=5):
        for e in range(epoch):
            self.feedforward(X)
            loss = self.loss_function(y, self.activations[-1])
            grads = self.backpropagation(X, y)
            self.gradient_decent(grads)
            print(f"Epoch {e + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.activations[-1], axis=1)

# Example usage
X = np.random.rand(100, 5)
y_labels = np.random.choice([0, 1], size=(100,))
y = np.zeros((100, 2))
y[np.arange(100), y_labels] = 1  # One-hot encoding

model = NeuralNetwork(inputSize=5, layers=[7, 5], outputSize=2)
model.train(X, y, epoch=10000)