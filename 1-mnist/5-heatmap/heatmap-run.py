"""
Heatmaps adapted for MNIST
MSIA 490: Deep Learning - Homework 1
4/26/17
Eric Chang
"""
import keras.datasets.mnist
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # without the .use() option, matplotlib doesn't play nice with AWS
import matplotlib.pyplot as plt
from heatmap import Heatmap


# TUNING PARAMETERS ###########################################################
activation_function = "relu"
cost_function = "cross_entropy"
init_method = "random_variance_normalized"

lr = 1e-05
epochs = 1000

visualize_every_n = 10
training_visuals = True
heatmap_visuals = True

subdirectory = "heatmaps-relu"

# directory structure:
#   filename/
#       subdirectory/
#           heat1/
#               heat00000.jpg, heat00001.jpg....
#           heat2/
#               heat00000.jpg, heat00001.jpg...
#           train00000.jpg, train00001.jpg...

# options
assert activation_function in ['sigmoid', 'tanh', 'relu', 'leaky_relu']
assert cost_function in ['mse', 'cross_entropy']
assert init_method in ["random", "random_variance_normalized",
                       "standard_normal", "standard_normal_variance_normalized"]
###############################################################################


def activation(x, method):
    """
    returns the input array with the specified activation applied
    """
    if method == "sigmoid":
        return 1.0 / (1.0 + np.e**-x)
    elif method == "tanh":
        return np.tanh(x)
    elif method == "relu" or method == "leaky_relu":
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
        gradient[gradient <= 0] = 0
        gradient[gradient != 0] = 1
        return gradient
    elif method == "leaky_relu":
        gradient = x.copy()
        gradient[gradient != 0] = 1
        gradient[gradient <= 0] = 0.01
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
        return 2 * np.random.rand(x, y).astype('float32').T - 1
    elif method == "random_variance_normalized":
        weights = 2 * np.random.rand(x, y).astype('float32').T - 1
        return weights / np.sqrt(x)
    elif method == "standard_normal":
        return np.random.randn(x, y).astype('float32').T
    elif method == "standard_normal_variance_normalized":
        weights = np.random.randn(x, y).astype('float32').T
        return weights / np.sqrt(x)


class Model:
    """
    A simple Model class to replace the keras.Model class in the heatmap code
    """
    def __init__(self, weights):
        self.weights = weights

    def predict(self, new_img):
        """
        given an input image of shape 28x28 or 784, returns class probabilites
        """
        assert isinstance(new_img, np.ndarray)

        # input from the heatmap could be flat, 28x28, or with RGB channels
        # so we need to flatten it to (784, )
        if not new_img.shape == (784,):
            if new_img.shape == (28, 28):
                new_img = new_img.reshape(784)
            elif new_img.shape == (1, 28, 28, 3):
                new_img = new_img[0, :, :, 0].reshape(784)  # choose an arbitrary channel - they're all the same in grayscale
            else:
                raise ValueError("Invalid image shape: " + str(new_img.shape))

        if not new_img.max() == 1:
            new_img = new_img / new_img.max()

        # perform forward pass
        (W1, W2, W3) = self.weights
        L1 = activation(W1.dot(new_img), activation_function)
        L2 = activation(W2.dot(L1), activation_function)
        L3 = activation(W3.dot(L2), activation_function)

        if not L3.shape == (10, ):
            L3 = L3[:, 0]
            assert L3.shape == (10, )

        return L3


if __name__ == "__main__":
    # extract filename and directory
    path = os.path.abspath(__file__)
    filename = path.split("/")[-1]
    directory = path[0:-len(filename)]

    # create directory structure:
    os.chdir(directory)
    os.makedirs(filename.split('.py')[0], exist_ok=True)
    os.chdir(filename.split('.py')[0])
    os.makedirs(subdirectory, exist_ok=True)

    (X, Y), (_, _) = keras.datasets.mnist.load_data()
    # Y = np.random.randint(0, 9, size=Y.shape) # Uncomment this line for
    # random labels extra credit
    X = X.astype('float32').reshape((len(X), -1)).T / 255.0  # 784 x 60000
    T = np.zeros((len(Y), 10), dtype='float32').T  # 10 x 60000
    for i in range(len(Y)):
        T[Y[i], i] = 1

    # Initialize weights
    if activation_function in ['relu', 'leaky_relu'] and "normalized" not in init_method:
        print("Not using normalized inputs with ReLU may result in exploding activations.")
    W1 = initialize_weights(784, 256, init_method)
    W2 = initialize_weights(256, 128, init_method)
    W3 = initialize_weights(128, 10, init_method)

    losses, accuracies, hw1, hw2, hw3, ma = [], [], [], [], [], []

    for i in range(epochs + 1):
        # Forward pass
        L1 = activation(W1.dot(X), activation_function)
        L2 = activation(W2.dot(L1), activation_function)
        L3 = activation(W3.dot(L2), activation_function)

        # Backward pass
        if cost_function == "mse":
            dW3 = (L3 - T) * gradient(L3, activation_function)
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

        # HEATMAP VISUALIZATIONS
        if heatmap_visuals and i % visualize_every_n == 0:
            images = [X[:, 7],   # 3
                      X[:, 4],   # 1
                      X[:, 10],  # messed up 3
                      X[:, 11]]  # weird 5

            # create directories for heatmaps
            if i == 0:
                for i in range(len(images)):
                    os.makedirs(os.path.join(subdirectory, "heat-" + str(i)))

            weights = (W1, W2, W3)
            model = Model(weights)
            heatmap = Heatmap(model)
            hmaps = []

            for img in images:
                img = img.reshape(28, 28)
                hmaps.append(heatmap.explain_prediction_heatmap(img), nmasks=(3, 4, 5, 7))

            for j, h in enumerate(hmaps):
                path = os.path.join(subdirectory, "heat-" + str(j), 'heat-%05d.png' % i)
                h.suptitle("Epoch: " + str(i))
                # h.text(x=0.5, y=0.5, s="Epoch: " + str(i))
                h.savefig(path)


        # TRAINING VISUALIZATIONS
        if training_visuals and i % visualize_every_n == 0:
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
