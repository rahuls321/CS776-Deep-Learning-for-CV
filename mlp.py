# from audioop import cross
import numpy as np

def relu(x, deriv=False):
    #ReLU activation function
    if(deriv):
        return np.where(x <= 0, 0, 1)
    return np.maximum(x, 0)

def softmax(x):
    '''
        softmax(x) = exp(x) / sum(exp(x))
    '''
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    # predict_class = np.argmax(prob, axis=1)
    return prob

def cross_entropy_loss(predicted, target):
    """
    predicted is the output from fully connected layer (num_examples x num_classes)
    target is labels (num_examples x 1)
    """
    # log_likelihood = -np.log(predicted[range(target.shape[0]), target])
    # print("target shape: ", target.shape)
    # print("predicted shape: ", np.log(predicted).shape)
    log_likelihood = np.multiply(target, np.log(predicted))
    loss = -np.sum(log_likelihood)/target.shape[0]
    return loss

class MLP(object):
    def __init__(self, input_size):
        self.weights = [0.1*np.random.randn(x, y) for x, y in zip(input_size[:-1], input_size[1:])]
        self.biases = [0.1*np.random.randn(1, x) for x in input_size[1:]]
        # print("Weights shape1: ", self.weights[0].shape)
        # print("Weights shape2: ", self.weights[1].shape)

        # print("Bias shape1: ", self.biases[0].shape)
        # print("Bias shape2: ", self.biases[1].shape)

    def feedforward(self, x):
        # print("X shape: ", x.shape)
        z1 = np.dot(x, self.weights[0]) + self.biases[0]
        # print("z1 shape: ", z1.shape)
        a1 = relu(z1)
        # print("a1 shape: ", a1.shape)
        z2 = np.dot(a1, self.weights[1]) + self.biases[1]
        # print("z2 shape: ", z2.shape)
        a2 = softmax(z2)
        # print("a2 shape: ", a2.shape)
        return z1, a1, z2, a2

    def backpropogation(self, X, y):

        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        # print("delta_w shape: ", delta_w[0].shape)
        # print("delta_b shape: ", delta_b[0].shape)

        z1, a1, z2, a2 = self.feedforward(X)
        loss = cross_entropy_loss(a2, y)
        # print("Loss: ", loss)
        error = a2 - y
        # print("error: ", error)

        #derivative of softmax = predicted_output - actual_output
        #For Hidden Layer
        delta1 = error
        delta_b[1] = np.sum(delta1, axis=0, keepdims=True)
        delta_w[1] = np.dot(a1.T, delta1)
        # print("delta_b[1].shape: ", delta_b[1].shape)
        # print("delta_w[1].shape: ", delta_w[1].shape)

        #For Input Layer
        deriv_relu = relu(z1, deriv=True)
        delta2 = np.dot(delta1, self.weights[1].T) * deriv_relu
        delta_b[0] = np.sum(delta2, axis=0, keepdims=True)
        delta_w[0] = np.dot(X.T, delta2)
        # print("delta_b[0].shape: ", delta_b[0].shape)
        # print("delta_w[0].shape: ", delta_w[0].shape)

        return loss, delta_b, delta_w

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output, axis=-1))

    def predict(self, X, labels):
        preds = np.array([])
        for x in X:
            _, _, _, outs = self.feedforward(x)
            preds = np.append(preds, labels[np.argmax(outs)])
        # preds = np.array([labels[int(p)] for p in preds])
        return preds
    
    def train(self, X, y, learning_rate=0.01, epochs=5, batch_size=100):
        for i in range(epochs):
            # Shuffle
            permutation = np.random.permutation(X.shape[0])
            x_train_shuffled = X[permutation]
            y_train_shuffled = y[permutation]
            losses=0.0

            for batch_idx in range(0, X.shape[0], batch_size):
                #Selecting batches of the training images
                batch_x = x_train_shuffled[batch_idx:batch_idx+batch_size]
                batch_y = y_train_shuffled[batch_idx:batch_idx+batch_size]

                # same shape as self.biases
                del_b = [np.zeros(b.shape) for b in self.biases]
                # same shape as self.weights
                del_w = [np.zeros(w.shape) for w in self.weights]

                loss, delta_b, delta_w = self.backpropogation(batch_x, batch_y)
                # losses+=loss
                del_b = [db + ddb for db, ddb in zip(del_b, delta_b)]
                del_w = [dw + ddw for dw, ddw in zip(del_w, delta_w)]

            self.weights = [w - (learning_rate/batch_size)*dw for w, dw in zip(self.weights, del_w)]
            self.biases = [b - (learning_rate/batch_size)*db for b, db in zip(self.biases, del_b)]

            # Evaluate performance
            # Training data
            _, _ , _,output = self.feedforward(X)
            train_acc = self.accuracy(y, output)
            train_loss = cross_entropy_loss(output, y)
            print("Epoch: %d Training loss: %.3f and Training accuracy: %.3f" %(i, train_loss, train_acc))

# X = np.array([[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]])

# y = np.array([0, 1, 2])
# y=y.reshape((3, 1))
# # print(y.shape)
# one_hot = np.zeros((y.shape[0], 3))
# one_hot[np.arange(y.shape[0]), y] = 1
# layer1 = MLP((4, 5, 3))
# print(layer1.feedforward(X))
# print(layer1.backpropogation(X, one_hot))
# print(layer1.train(X, one_hot, epochs=10))

def Model(X, y, labels):

    inp_feats=512
    num_hidden=64
    num_output=10
    epochs=20
    y=np.array(y)
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    
    model = MLP((inp_feats, num_hidden, num_output))
    model.train(X, y_one_hot, epochs=epochs)
    pred = model.predict(X, labels)
    print(model.accuracy(pred, y))
