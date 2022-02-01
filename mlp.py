# from audioop import cross
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

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
    # print("exp: ", exp_values)
    prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    # predict_class = np.argmax(prob, axis=1)
    # prob = prob.reshape(-1, 10, 1)
    # print("prob: ", prob)
    # exit()
    return prob

def cross_entropy_loss(predicted, target):
    """
    predicted (batch_size x num_classes)
    target is labels (batch_size x 1)
    """
    log_likelihood = -np.log(predicted[range(target.shape[0]), target])
    # print("target shape: ", target.shape)
    # print("predicted shape: ", np.log(predicted).shape)
    # exit()
    # log_likelihood = np.multiply(target, np.log(predicted))
    loss = -np.sum(log_likelihood)/target.shape[0]
    # print("Loss: ", np.sum(loss))
    # exit()
    return loss

def get_one_hot_vector(y):
    """
    generating one hot vector for class labels
    """
    y=np.array(y)
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    # y_one_hot = y_one_hot.reshape(-1, 10, 1)
    return y_one_hot

class MLP(object):
    def __init__(self, input_size):
        self.weights = [np.random.normal(0, 2/x, (y, x)) for x, y in zip(input_size[:-1], input_size[1:])]
        self.biases = [np.zeros((x, 1)) for x in input_size[1:]]
        print("Weights shape1: ", self.weights[0].shape)
        print("Weights shape2: ", self.weights[1].shape)

        print("Bias shape1: ", self.biases[0].shape)
        print("Bias shape2: ", self.biases[1].shape)

    def feedforward(self, x):
        # print("X shape: ", x.shape)
        # print("Weights shape1: ", self.weights[0].shape)
        # print("Bias shape1: ", self.biases[0].shape)
        # exit()
        z1 = np.dot(self.weights[0], x) + self.biases[0][:, :x.shape[1]]
        # print("z1 shape: ", z1.shape)/
        a1 = relu(z1)
        # print("a1 shape: ", a1.shape)
        z2 = np.dot(self.weights[1], a1) + self.biases[1][:, :a1.shape[1]]
        # print("z2 shape: ", z2.shape)
        a2 = softmax(z2.T)
        # print("a2 shape: ", a2.shape)
        return z1, a1, z2, a2

    def backpropagation(self, X, y):

        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        # print("delta_w shape: ", delta_w[0].shape)
        # print("delta_b shape: ", delta_b[0].shape)

        z1, a1, z2, a2 = self.feedforward(X)
        # exit()
        y = y.argmax(axis=1)
        loss = cross_entropy_loss(a2, y)
        # print("Loss: ", loss)
        a2[range(y.shape[0]), y] -= 1
        # a2 /= a2.shape[0]
        error = a2
        # print("a2: ", a2)
        # print("y: ", y)
        # print("error: ", error.shape)
        # exit()
        # print("a1: ", a1.shape)

        #derivative of softmax = predicted_output - actual_output
        #For Hidden Layer
        delta1 = error.T
        delta_b[1] = delta1 # np.sum(delta1, axis=0, keepdims=True)
        delta_w[1] = np.dot(delta1, a1.T)
        # print("delta_b[1].shape: ", delta_b[1].shape)
        # print("delta_w[1].shape: ", delta_w[1].shape)
        # exit()

        # print("delta1 shape", delta1.shape)
        # print("self.weights[1]: ", self.weights[1].T.shape)
        
        
        #For Input Layer
        deriv_relu = relu(z1, deriv=True)
        # print("deriv_relu shape: ", deriv_relu.shape)
        delta2 = np.dot(self.weights[1].T, delta1) * deriv_relu
        delta_b[0] = delta2
        # print("delta2 shape: ", delta2.shape)
        # X = X.reshape(1, 512)
        delta_w[0] = np.dot(delta2, X.T)
        # print("X shape: ", X.shape)
        
        # print("delta_b[0].shape: ", delta_b[0].shape)
        # print("delta_w[0].shape: ", delta_w[0].shape)
        # exit()
        return loss, delta_b, delta_w

    def evaluate(self, X, y):

        #Here y is single value and x is same as input
        count = 0
        for x, _y in zip(X, y):
            # postion of maximum value is the predicted label
            # print("pred: ", np.argmax(p))
            # print("pred: ", p)
            # print("y: ", np.argmax(_y))
            # print("y: ", _y)
            # print("y: ", np.argmax(_y))
            _, _ , _,output = self.feedforward(np.array([x]).T)
            # print("output: ",  output)
            # print("output: ",  np.argmax(output))
            # exit()
            if np.argmax(output) == np.argmax(_y):
                count += 1
        return float(count) / X.shape[0]

    # def predict(self, X, labels):
    #     preds = np.array([])
    #     for x in X:
    #         _, _, _, outs = self.feedforward(x)
    #         preds = np.append(preds, labels[np.argmax(outs)])
    #     # preds = np.array([labels[int(p)] for p in preds])
    #     return preds
    
    def train(self, X, y, X_test, y_test, learning_rate=0.01, epochs=5, batch_size=100):
        history_training_loss, history_training_acc, history_test_acc=[], [], []
        for epoch in range(epochs):
            # Shuffle
            permutation = np.random.permutation(X.shape[0])
            x_train_shuffled = X[permutation]
            y_train_shuffled = y[permutation]
            

            # same shape as self.biases
            del_b = [np.zeros(b.shape) for b in self.biases]
            # same shape as self.weights
            del_w = [np.zeros(w.shape) for w in self.weights]
            flag=0
            loss_min=9999

            for batch_idx in range(0, X.shape[0], batch_size):
                #Selecting batches of the training images
                batch_x = x_train_shuffled[batch_idx:batch_idx+batch_size]
                batch_y = y_train_shuffled[batch_idx:batch_idx+batch_size]

                # print("Batch_x: ", batch_x.shape)
                # print("Batch_y: ", batch_y.shape)
                # losses=0.0
                #Performing Backpropagation
                loss, delta_b, delta_w = self.backpropagation(batch_x.T, batch_y)
                loss = abs(loss)
                # print("Delta_b_0: ", delta_b[0].shape)
                # print("Delta_w_0: ", delta_w[0].shape)
                # print("Delta_b_1: ", delta_b[1].shape)
                # print("Delta_w_1: ", delta_w[1].shape)
                # # exit()
                #Maintaining the original shape according to batch size when input is less than given batch size
                if delta_b[0].shape[1] < batch_size:
                    # print(delta_b[0].shape)
                    size_left = batch_size - delta_b[0].shape[1]
                    delta_b[0] = np.pad(delta_b[0], ((0,0), (0,size_left)))

                if delta_b[1].shape[1] < batch_size:
                    # print(delta_b[1].shape)
                    size_left = batch_size - delta_b[1].shape[1]
                    delta_b[1] = np.pad(delta_b[1], ((0,0), (0, size_left)))

                # losses+=loss
                
                # print("Del_b_0: ", del_b[0].shape)
                # print("Del_b_1: ", del_b[1].shape)
                # print("Del_w_0: ", del_w[0].shape)
                # print("Del_w_1: ", del_w[1].shape)
                if flag==0:
                    del_b = [delta_b[0], delta_b[1]]
                    del_w = [delta_w[0], delta_w[1]]
                    flag=1
                else:
                    #It will encounter only when there's minimum loss is getting as compared to previous batches
                    if loss < loss_min:
                        loss_min=loss
                        for i, (db, dw)in enumerate(zip(delta_b, delta_w)):
                            # print("db.shape: ", db.shape)
                            # print("dw.shape: ", dw.shape)
                            # print(del_b[i])
                            del_b[i] += db
                            del_w[i] += dw
                # del_b = [db + ddb for db, ddb in zip(del_b, delta_b)]
                # del_w = [dw + ddw for dw, ddw in zip(del_w, delta_w)]
            
            # print("Del_b_0: ", del_b[0].shape)
            # print("Del_w_0: ", del_w[0].shape)
            # print("Del_b_1: ", del_b[1].shape)
            # print("Del_w_1: ", del_w[1].shape)

            self.weights = [w - (learning_rate/batch_size)*dw for w, dw in zip(self.weights, del_w)]
            self.biases = [b - (learning_rate/batch_size)*db for b, db in zip(self.biases, del_b)]

            # print("Weights shape1: ", self.weights[0].shape)
            # print("Weights shape2: ", self.weights[1].shape)

            # print("Bias shape1: ", self.biases[0].shape)
            # print("Bias shape2: ", self.biases[1].shape)

            # Evaluate performance
            train_acc = self.evaluate(X, y)
            test_acc = self.evaluate(X_test, y_test)
            history_training_loss.append(loss_min)
            history_training_acc.append(train_acc)
            history_test_acc.append(test_acc)
            print("Epoch: %d Training loss: %.3f Training accuracy: %.2f" %(epoch, loss_min, train_acc*100))
        return np.array(history_training_acc), np.array(history_training_loss), np.array(history_test_acc)

def Model(X_train, y_train, X_test, y_test, augmented=False):

    inp_feats=512
    num_hidden=64
    num_output=10
    epochs=500
    batch_size=256
    learning_rate=0.01
    print("y_train: ",y_train.shape)
    # y_train = get_one_hot_vector(y_train)
    for batch_size in [32, 64, 128, 256]:
        print("batch-size: ", batch_size)
        for learning_rate in [0.1, 0.01, 0.001]:
            ##print("batch-size: ", batch_size)
            print("learning_rate: ", learning_rate)
            model = MLP((inp_feats, num_hidden, num_output))
            total_training_acc, total_training_loss, total_testing_acc = model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
            # acc = model.evaluate(X_test, y_test)
            print("Testing Accuracy or Performance Measure in percent: %.2f" %(total_testing_acc[np.argmax(total_training_acc)]*100))
            plt.figure(figsize= (8,8))
            plt.plot(np.arange(epochs), total_training_acc, label='Total_trainig_acc')
            plt.plot(np.arange(epochs), total_training_loss, label='Total_training_loss')
            # plt.plot(np.arange(epochs), total_testing_acc, label='Total_training_acc')
            plt.xlabel("Epochs")
            plt.ylabel("Training Accuracy and Loss")
            plt.title("Loss/Accuracy vs Epochs - "+"batch_size-"+str(batch_size)+"-learning-rate-"+str(learning_rate))
            plt.legend()
            if(augmented):
                plt.savefig("./output/augmented/loss-accuracy-graph-"+str(batch_size)+"-"+str(learning_rate)+".jpg")
            else: plt.savefig("./output/loss-accuracy-graph-"+str(batch_size)+"-"+str(learning_rate)+".jpg")
    return model
