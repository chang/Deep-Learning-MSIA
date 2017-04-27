# Homework 1 starter code: Simple neural network test test
import keras.datasets.mnist
import numpy as np
import os
import heatmap as hm
from heatmap import Heatmap

########################################################################## 
# YOU CAN IGNORE THIS STUFF - I MADE THESE CHANGES FOR MY OWN WORKFLOW
ON_SERVER = True

# import matplotlib, bypassing graphics display if on server
if ON_SERVER:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# extract filename and directory
path = os.path.abspath(__file__)
filename = path.split("/")[-1]
directory = path[0:-len(filename)]

os.chdir(directory)
os.makedirs(filename.split('.py')[0], exist_ok=True)
os.chdir(filename.split('.py')[0])
##########################################################################

# TUNING PARAMETERS
activation_function = "relu"
cost_function = "cross_entropy"
init_method = "random_variance_normalized"
lr = 1e-05
original_viz = False
heatmap_viz = True

subdirectory = "final"

# parameter options
assert activation_function in ['sigmoid', 'tanh', 'relu', 'leaky_relu']
assert cost_function in ['mse', 'cross_entropy']
assert init_method in ["random", "random_variance_normalized",
                       "standard_normal", "standard_normal_variance_normalized"]

os.makedirs(subdirectory, exist_ok=True)


def activation(x, method):
    """
    returns an array with the specified activation applied
    """
    if method == "sigmoid":
        return 1.0 / (1.0 + np.e**-x)
    elif method == "tanh":
        return np.tanh(x)
    elif method == "relu" or method == "leaky_relu":
        if init_method not in ["standard_normal_variance_normalized",
                               "random_variance_normalized"]:
            print("Failing to normalize variance when using ReLU may result in exploding activations.")
        return np.maximum(0, x)


def gradient(x, method):
    """
    returns the gradient array of the specified activation
    """
    if method == "sigmoid":
        return x * (1 - x)
    elif method == "tanh":
        return 1 - x**2
    elif method == "relu":
        gradient = x.copy()
        gradient[gradient != 0] = 1
        return gradient
    elif method == "leaky_relu":
        gradient = x.copy()
        gradient[gradient != 0] = 1
        gradient[gradient == 0] = 0.01
        return gradient        


def initialize_weights(x, y, method):
    """
    initializes a vector of shape (x, y)
    with weights created using various methods.
    "random": random weights in range [-1, 1]
    "standard_normal": weights sampled from a standard normal distribution
    "standard_normal_variance_normalized": weights sampled from a standard normal distribution, then variance normalized
    """
    if method == "random":
        weights = 2 * np.random.rand(x, y).astype('float32').T - 1
    elif method == "random_variance_normalized":
        weights = (2 * np.random.rand(x, y).astype('float32').T - 1) / np.sqrt(x)
    elif method == "standard_normal":
        weights = np.random.randn(x, y).astype('float32').T
    elif method == "standard_normal_variance_normalized":
        weights = (np.random.randn(x, y).astype('float32').T) / np.sqrt(x)
    else:
        raise ValueError("Weight initialization method not valid.")

    return weights


class Model:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, new_img):
        """
        given an input image of shape 28x28 or 784
        returns class probabilites
        """
        assert isinstance(new_img, np.ndarray)

        if not new_img.shape == (784,):
            
            if new_img.shape == (28, 28):
                new_img = new_img.reshape(784)
            elif new_img.shape == (1, 28, 28, 3):
                new_img = new_img[0, :, :, 0].reshape(784)  # just choose an arbitrary channel
            else:
                raise ValueError("Invalid image shape", new_img.shape)

        if not new_img.max() == 1:
            new_img = new_img / new_img.max()

        # perform forward pass
        (W1, W2, W3) = self.weights
        L1 = activation(W1.dot(new_img), activation_function)
        L2 = activation(W2.dot(L1), activation_function)
        L3 = activation(W3.dot(L2), activation_function)
        
        if not L3.shape == (10, ):
            L3 = L3[:, 0]

        # RETURN A NUMPY ARRAY OF SHAPE (10, )
        return L3


if __name__ == "__main__":
    (X, Y), (_, _) = keras.datasets.mnist.load_data()

    # Y = np.random.randint(0, 9, size=Y.shape) # Uncomment this line for
    # random labels extra credit
    X = X.astype('float32').reshape((len(X), -1)).T / 255.0  # 784 x 60000
    T = np.zeros((len(Y), 10), dtype='float32').T  # 10 x 60000
    for i in range(len(Y)):
        T[Y[i], i] = 1

    # Initialize weights
    W1 = initialize_weights(784, 256, init_method)
    W2 = initialize_weights(256, 128, init_method)
    W3 = initialize_weights(128, 10, init_method)

    losses, accuracies, hw1, hw2, hw3, ma = [], [], [], [], [], []

    for i in range(1001):  # Do not change this, we will compare performance at 1000 epochs
        # Forward pass
        L1 = activation(W1.dot(X), activation_function)
        L2 = activation(W2.dot(L1), activation_function)
        L3 = activation(W3.dot(L2), activation_function)

        # Backward pass
        if cost_function == "mse":
            dW3 = (L3 - T) * gradient(L3, activation_function)  # * L3 * (1 - L3)
        elif cost_function == "cross_entropy":
            dW3 = (L3 - T)
        dW2 = W3.T.dot(dW3) * gradient(L2, activation_function)
        dW1 = W2.T.dot(dW2) * gradient(L1, activation_function)

        # Update
        W3 -= lr * np.dot(dW3, L2.T)
        W2 -= lr * np.dot(dW2, L1.T)
        W1 -= lr * np.dot(dW1, X.T)

        loss = np.sum((L3 - T)**2) / len(T.T)
        print("[%04d] MSE Loss: %0.6f" % (i, loss))
        losses.append(loss)

        # HEATMAP
        if heatmap_viz and i % 10 == 0:
            img = X[:, 0].reshape(28, 28)  # test heatmap on first image

            images = X
            activations = (L1, L2, L3)
            weights = (W1, W2, W3)

            # bundle into our class Model
            model = Model(weights)
            heatmap = Heatmap(model)
            h = heatmap.explain_prediction_heatmap(img, nmasks=(4, 5))

        if original_viz and i % 10 == 0:
            predictions = np.zeros(L3.shape, dtype='float32')
            for j, m in enumerate(np.argmax(L3.T, axis=1)):
                predictions[m, j] = 1
            acc = np.sum(predictions * T)
            accpct = 100 * acc / X.shape[1]
            accuracies.append(accpct)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            testi = np.random.choice(range(60000))
            ax1.imshow(X.T[testi].reshape(28, 28), cmap='gray')
            ax1.set_xticks([]), ax1.set_yticks([])
            cls = np.argmax(L3.T[testi])
            ax1.set_title("Prediction: %d confidence=%0.2f" %
                          (cls, L3.T[testi][cls] / np.sum(L3.T[testi])))

            ax2.plot(losses, color='blue')
            ax2.set_title("Loss"), ax2.set_yscale('log')
            ax3.plot(accuracies, color='blue')
            ax3.set_ylim([0, 100])
            # Aim for 90% accuracy in 200 epochs
            ax3.axhline(90, color='red', linestyle=':')
            ax3.set_title("Accuracy: %0.2f%%" % accpct)
            plt.savefig(os.path.join(subdirectory, 'train-acc-%05d.png' % i))
            plt.show(), plt.close()

            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
                  (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(10, 10))
            ax1.imshow(np.reshape(L1.mean(axis=1), (16, 16,)), cmap='gray', interpolation='none'), ax1.set_title(
                'L1 $\mu$=%0.2f $\sigma$=%0.2f' % (L1.mean(), L1.std()))
            ax2.imshow(np.reshape(L2.mean(axis=1), (16, 8,)),  cmap='gray', interpolation='none'), ax2.set_title(
                'L2 $\mu$=%0.2f $\sigma$=%0.2f' % (L2.mean(), L2.std()))
            ax3.imshow(np.reshape(L3.mean(axis=1), (10, 1,)),  cmap='gray', interpolation='none'), ax3.set_title(
                'L3 $\mu$=%0.2f $\sigma$=%0.2f' % (L3.mean(), L3.std())), ax3.set_xticks([])
            activations = np.concatenate(
                (L1.flatten(), L2.flatten(), L3.flatten()))
            ma.append(activations.mean())
            ax4.plot(ma, color='blue'), ax4.set_title(
                "Activations $\mu$: %0.2f $\sigma$=%0.2f" % (ma[-1], activations.std()))
            ax4.set_ylim(0, 1)

            ax5.imshow(np.reshape(W1.mean(axis=0), (28, 28,)), cmap='gray', interpolation='none'), ax5.set_title(
                'W1 $\mu$=%0.2f $\sigma$=%0.2f' % (W1.mean(), W1.std()))
            ax6.imshow(np.reshape(W2.mean(axis=0), (16, 16,)), cmap='gray', interpolation='none'), ax6.set_title(
                'W2 $\mu$=%0.2f $\sigma$=%0.2f' % (W2.mean(), W2.std()))
            ax7.imshow(np.reshape(W3.mean(axis=0), (16, 8, )), cmap='gray', interpolation='none'), ax7.set_title(
                'W3 $\mu$=%0.2f $\sigma$=%0.2f' % (W3.mean(), W3.std())), ax7.set_xticks([])
            ax8.plot(accuracies, color='blue'), ax8.set_title(
                "Accuracy: %0.2f%%" % accpct), ax8.set_ylim(0, 100)

            uw1, uw2, uw3 = np.dot(dW1, X.T), np.dot(dW2, L1.T), np.dot(dW3, L2.T)
            hw1.append(lr * np.abs(uw1).mean()), hw2.append(lr *
                                                            np.abs(uw2).mean()), hw3.append(lr * np.abs(uw3).mean())
            ax9.imshow(np.reshape(uw1.sum(axis=0),  (28, 28,)), cmap='gray', interpolation='none'), ax9.set_title(
                '$\Delta$W1: %0.2f E-5' % (1e5 * lr * np.abs(uw1).mean()), color='r')
            ax10.imshow(np.reshape(uw2.sum(axis=0), (16, 16,)), cmap='gray', interpolation='none'), ax10.set_title(
                '$\Delta$W2: %0.2f E-5' % (1e5 * lr * np.abs(uw2).mean()), color='g')
            ax11.imshow(np.reshape(uw3.sum(axis=0), (16, 8, )), cmap='gray', interpolation='none'), ax11.set_title(
                '$\Delta$W3: %0.2f E-5' % (1e5 * lr * np.abs(uw3).mean()), color='b'), ax11.set_xticks([])
            ax12.plot(hw1, color='r'), ax12.plot(hw2, color='g'), ax12.plot(
                hw3, color='b'), ax12.set_title('Weight update magnitude')
            ax12.legend(loc='upper right'), ax12.set_yscale('log')

            plt.suptitle(
                "Weight and update visualization ACC: %0.2f%% LR=%0.8f" % (accpct, lr))
            plt.savefig(os.path.join(subdirectory, 'train-%05d.png' % i))
            plt.show(), plt.close()
