import math
import numpy as np
import matplotlib.pyplot as plt


def clear_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def get_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clear_input = list(map(clear_data, input_data))
        f.close()
    return clear_input


def readFile(dataset_address):
    input_data = get_data(dataset_address)
    input_np = np.array(input_data)
    return input_np


def gen_data(X, d, k):
    num_samples, num_features = X.shape
    array_list = []
    if d == 0:
        return np.array([[] for _ in range(num_samples)])
    else:
        for i in range(1, d+1):
            array_list.append(np.sin(X * k * i)**2)
        data = np.concatenate(array_list, axis=1)
        return data


def homogenize_data(data):
    temp = np.ones((data.shape[0], data.shape[1] + 1))
    temp[:, 1:] = data
    return temp


def squared_loss(X, y, w):
    num_points, num_features = X.shape
    y_pred = X @ w
    loss_vec = (y_pred - y)**2
    loss = np.sum(loss_vec)
    grad_vec = -2 * np.reshape(y - y_pred, (num_points, 1)) * X
    grad = np.sum(grad_vec, axis=0)
    return loss, grad


def numericalgrad(funObj, X, y, w, epsilon):
    m = len(w)
    grad = np.zeros(m)
    for i in range(m):
        wp = np.copy(w)
        wn = np.copy(w)
        wp[i] = w[i] + epsilon
        wn[i] = w[i] - epsilon
        grad[i] = (funObj(X, y, wp)[0] - funObj(X, y, wn)[0])/(2 * epsilon)
    return grad


def get_preds(X, w):
    y_pred = X @ w
    return y_pred


def gradient_descent(X, y, loss_fun, lr, thres):
    num_points, num_features = X.shape
    w = np.zeros(num_features)
    iter = 0
    while True:
        loss, grad = loss_fun(X, y, w)
        print("Iteration: " + str(iter) + " ,Loss: " + str(loss) + " , grad_norm:" +str(np.linalg.norm(grad, ord=2)))
        if np.linalg.norm(grad, ord=2) < thres:
            break
        w = w - lr * grad
        iter = iter + 1
    return w

	
def gd_with_backtracking_v4(X, y, funObj, verbosity, gamma, thres):
    num_points, num_features = X.shape
    w = np.zeros(num_features)
    [f, g] = funObj(X, y, w)
    alpha = 1 / np.linalg.norm(g, 2)
    funEvals = 1
    funVals = []
    iterations = 1
    backtrackings = 0
    while (1):
        funEvals = funEvals + 1
        [f, g] = funObj(X, y, w)
        gg = np.linalg.norm(g, 2) ** 2
        while (1):
            w_new = w - (alpha * g)
            [f_new, g_new] = funObj(X, y, w_new)
            funEvals = funEvals + 1
            if f_new > (f - gamma * alpha * (np.linalg.norm(g, 2) ** 2)):
                alpha = (alpha * alpha * gg) / (2 * (f_new + gg * alpha - f))
                backtrackings = backtrackings + 1
            else:
                break
        w = w - (alpha * g)
        f_new, _ = funObj(X, y, w)
        optCond = np.linalg.norm(g, ord=2)
        if verbosity > 0:
            print(iterations, f, optCond, funEvals, backtrackings, alpha)
        alpha = min(1, 2 * (f - f_new) / gg)
        iterations += 1
        if (optCond < thres) or (alpha == 0):
            break
        funVals.append(f)
    return w


def main():
    print('START Q1_AB\n')
    #Load data for Q1_B
    train_data = readFile('datasets/Q1_B_train.txt')
    X_train_np = train_data[:, :-1].astype('float')
    Y_train_np = train_data[:, -1].astype('float')
    #Generating data from the basis functions
    
    for k in range(1, 11):
        #plt.figure()
        for d in range(0, 7):
            X_input = gen_data(X_train_np, k=k, d=d)
            
            #Homogenizing data by adding one's column to the data matrix.
            X_input = homogenize_data(X_input)
            num_points, num_features = X_input.shape
            w = np.zeros(num_features)

            #Asserting that the squared loss gradient computation is accurate
            loss, grad = squared_loss(X_input, Y_train_np, w)
            num_grad = numericalgrad(squared_loss, X_input, Y_train_np, w, 1e-5)
            assert np.linalg.norm(grad-num_grad)<1e-2, "Gradient computed is not matching the numerical gradient"

            #Standard Gradient Descent Function
            #w_learned = gradient_descent(X_input, Y_train_np, squared_loss, 1e-4, 1e-5)

            #Gradient Descent with back tracking for faster model convergence and loss reduction
            w_learned = gd_with_backtracking_v4(X_input, Y_train_np, squared_loss, 0, 1e-4, 1e-5)
            
            y_preds = get_preds(X_input, w_learned)
            #plt.subplot(7, 1, d+1)
            plt.figure()
            plt.plot(X_train_np, Y_train_np, color='blue', marker='o', linestyle='None',
                   linewidth=2, markersize=10, label='ground_truth')
            plt.plot(X_train_np, y_preds, color='orange', marker='x', linestyle='None',
                    linewidth=2, markersize=10, label='predicted')
            plt.title('k= '+str(k) + " , " + "d= " + str(d))
            plt.legend()
            plt.savefig("Q1B_Visualization_k=" + str(k) + "," + "d=" + str(d) + ".jpg")
    #plt.savefig("Visualization of learned functions")
    #plt.show()
    print('END Q1_AB\n')


if __name__ == "__main__":
    main()
	