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


def homogenize_data(data):
    temp = np.ones((data.shape[0], data.shape[1] + 1))
    temp[:, 1:] = data
    return temp


def weighted_squared_loss(X, y, w, query, gamma=0.204):
	sample_weights = np.exp(-np.linalg.norm(X - query, axis=1)**2/(2 * gamma**2)).reshape((-1, 1))
	num_points, num_features = X.shape
	y_pred = X @ w
	loss_vec = sample_weights * (y_pred.reshape(-1, 1) - y.reshape(-1, 1))**2
	loss = np.sum(loss_vec)
	grad_vec = -2 * sample_weights * np.reshape(y.reshape(-1, 1) - y_pred.reshape(-1, 1), (num_points, 1)) * X
	grad = np.sum(grad_vec, axis=0)
	return loss, grad


def compute_squared_loss(y, y_pred):
    loss = np.sum((y - y_pred)**2)
    return loss


def numericalgrad(funObj, X, y, w, query, epsilon):
    m = len(w)
    grad = np.zeros(m)
    for i in range(m):
        wp = np.copy(w)
        wn = np.copy(w)
        wp[i] = w[i] + epsilon
        wn[i] = w[i] - epsilon
        grad[i] = (funObj(X, y, wp, query)[0] - funObj(X, y, wn, query)[0])/(2 * epsilon)
    return grad


def get_preds(X, w):
    y_pred = X @ w
    return y_pred


def gradient_descent(X, y, query, loss_fun, lr, thres):
    num_points, num_features = X.shape
    w = np.zeros(num_features)
    iter = 0
    while True:
        loss, grad = loss_fun(X, y, w, query)
        print("Iteration: " + str(iter) + " ,Loss: " + str(loss) + " , grad_norm:" +str(np.linalg.norm(grad, ord=2)))
        if np.linalg.norm(grad, ord=2) < thres:
            break
        w = w - lr * grad
        iter = iter + 1
    return w

	
def gd_with_backtracking_v4(X, y, query, funObj, verbosity, gamma, thres):
    num_points, num_features = X.shape
    w = np.zeros(num_features)
    [f, g] = funObj(X, y, w, query)
    alpha = 1 / np.linalg.norm(g, 2)
    funEvals = 1
    funVals = []
    iterations = 1
    backtrackings = 0
    while (1):
        funEvals = funEvals + 1
        [f, g] = funObj(X, y, w, query)
        gg = np.linalg.norm(g, 2) ** 2
        while (1):
            w_new = w - (alpha * g)
            [f_new, g_new] = funObj(X, y, w_new, query)
            funEvals = funEvals + 1
            if f_new > (f - gamma * alpha * (np.linalg.norm(g, 2) ** 2)):
                alpha = (alpha * alpha * gg) / (2 * (f_new + gg * alpha - f))
                backtrackings = backtrackings + 1
            else:
                break
        w = w - (alpha * g)
        f_new, _ = funObj(X, y, w, query)
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
	print('START Q2_D\n')
	#Load data for Q1_B
	train_data = readFile('datasets/Q1_B_train.txt')
	X_train_np = train_data[:20, :-1].astype('float')
	Y_train_np = train_data[:20, -1].astype('float')

	#Load test data for Q1_C
	test_data = readFile('datasets/Q1_C_test.txt')
	X_test_np = test_data[:, :-1].astype('float')
	Y_test_np = test_data[:, -1].astype('float')

	#Homogenizing data by adding one's column to the data matrix.
	trn_input = homogenize_data(X_train_np)

	#Homogenizing data by adding one's column to the data matrix.
	tst_input = homogenize_data(X_test_np)

	y_preds = []
	for i in range(trn_input.shape[0]):
		query = trn_input[i]
		num_points, num_features = trn_input.shape
		w = np.zeros(num_features)

		#Asserting that the squared loss gradient computation is accurate
		loss, grad = weighted_squared_loss(trn_input, Y_train_np, w, query=query)
		num_grad = numericalgrad(weighted_squared_loss, trn_input, Y_train_np, w, query, 1e-5)
		assert np.linalg.norm(grad-num_grad)<1e-2, "Gradient computed is not matching the numerical gradient"

		#Standard Gradient Descent Function
		#w_learned = gradient_descent(X_input, Y_train_np, query, squared_loss, 1e-4, 1e-5)

		#Gradient Descent with back tracking for faster model convergence and loss reduction
		w_learned = gd_with_backtracking_v4(trn_input, Y_train_np,query, weighted_squared_loss, 0, 1e-4, 1e-5)
		
		y_pred = get_preds(query, w_learned)
		y_preds.append(y_pred)

	plt.figure()
	plt.plot(X_train_np, Y_train_np, color='blue', marker='o', linestyle='None',
			linewidth=2, markersize=10, label='ground_truth')
	plt.plot(X_train_np, y_preds, color='orange', marker='x', linestyle='None',
			linewidth=2, markersize=10, label='predicted')
	plt.title('Q2D_Learned_Function')
	plt.legend()
	plt.savefig("Q2D_Visualization.jpg")

	y_preds = []
	for i in range(tst_input.shape[0]):
		query = tst_input[i]
		num_points, num_features = trn_input.shape
		w = np.zeros(num_features)

		#Asserting that the squared loss gradient computation is accurate
		loss, grad = weighted_squared_loss(trn_input, Y_train_np, w, query=query)
		num_grad = numericalgrad(weighted_squared_loss, trn_input, Y_train_np, w, query, 1e-5)
		assert np.linalg.norm(grad-num_grad)<1e-2, "Gradient computed is not matching the numerical gradient"

		#Standard Gradient Descent Function
		#w_learned = gradient_descent(X_input, Y_train_np, query, squared_loss, 1e-4, 1e-5)

		#Gradient Descent with back tracking for faster model convergence and loss reduction
		w_learned = gd_with_backtracking_v4(trn_input, Y_train_np,query, weighted_squared_loss, 0, 1e-4, 1e-5)
		
		y_pred = get_preds(query, w_learned)
		y_preds.append(y_pred)

	print("Squared loss on test dataset is " + str(compute_squared_loss(Y_test_np, y_preds)))

	print('END Q2_D\n')


if __name__ == "__main__":
    main()
	