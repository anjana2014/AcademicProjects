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


def logistic_loss(X, y, w):
	num_points, num_features = X.shape
	margin_vector = X @ w.reshape(-1, 1)
	output_mask = np.array((1 - y)).reshape((-1, 1))
	loss_vec = (output_mask * margin_vector) + np.logaddexp(0, -1 * margin_vector)
	loss = np.sum(loss_vec, axis=0)
	p_1 = 1 / (1 + np.exp(-1 * margin_vector)).reshape((-1, 1))
	grad_vec = (p_1 - y.reshape((-1, 1))) * X
	grad = np.sum(grad_vec, axis=0)
	return loss, grad


def cal_accuracy(targets, pred):
    acc = (len(np.where(targets == pred)[0]) / len(pred)) * 100
    return acc


def get_preds(X, w):
	margin_vec = X @ w
	y_pred = np.where(margin_vec>0, 1, 0)
	return y_pred


def get_probs(X, w):
	margin_vec = X @ w
	p_1 = 1 / (1 + np.exp(-1 * margin_vec)).reshape((-1, 1))
	p_0 = 1 - p_1
	return np.concatenate([p_0, p_1], axis=1)


def gradient_descent(X, y, loss_fun, lr, thres, max_iterations=10000):
	num_points, num_features = X.shape
	w = np.zeros(num_features)
	iter = 0
	while True:
		loss, grad = loss_fun(X, y, w)
		print("Iteration: " + str(iter) + " ,Loss: " + str(loss) + " , grad_norm:" +str(np.linalg.norm(grad, ord=2)))
		w = w - lr * grad
		iter = iter + 1
		if (np.linalg.norm(grad, ord=2) < thres) or (iter > max_iterations):
			break
	return w

	
def gd_with_backtracking_v4(X, y, funObj, verbosity, gamma, thres, max_iterations = 20000):
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
        if (optCond < thres) or (alpha == 0) or (iterations > max_iterations):
            break
        funVals.append(f)
    return w


#Principal Component Analysis for Projections
class pca_algo:
	def __init__(self):
		self.eigen_vectors = None
		self.eigen_values = None

	def fit(self, data):
		#n X p
		# Zeroing out the mean
		self.mean = np.mean(data, axis=0).reshape((1, -1))
		data = np.transpose(np.subtract(data, np.mean(data, axis=0)))
		#np.transpose(np.subtract(np.transpose(data), np.mean(data, axis=1)))
		# Calculating the Covariance Matrix
		covariance_matrix = data @ np.transpose(data)
		eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
		sort_index = np.argsort(eigen_values)[::-1]
		self.eigen_values = np.sort(eigen_values)[::-1]
		self.eigen_vectors = eigen_vectors[:, sort_index]

	def fit_transform(self, data, k):
        # Zeroing out the mean
		self.mean = np.mean(data, axis=0).reshape((1, -1))
		data = np.transpose(np.subtract(data, np.mean(data, axis=0)))
		# Calculating the Covariance Matrix
		covariance_matrix = data @ np.transpose(data)
		eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
		sort_index = np.argsort(eigen_values)[::-1]
		self.eigen_values = np.sort(eigen_values)[::-1]
		self.eigen_vectors = eigen_vectors[:, sort_index]
		transformed_data = np.transpose(self.eigen_vectors[:, 0:k]) @ data
		return np.transpose(transformed_data)

	def transform(self, data, k):
		data = np.transpose(np.subtract(data, self.mean))
		transformed_data = np.transpose(self.eigen_vectors[:, 0:k]) @ data
		return np.transpose(transformed_data)


class StandardScaling:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, data):
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        transform_data = np.subtract(data, self.mean)
        transform_data = np.divide(transform_data, self.std)
        return transform_data

    def transform(self, data):
        transform_data = np.subtract(data, self.mean)
        transform_data = np.divide(transform_data, self.std)
        return transform_data


def main():
	print('START Q3_AB\n')
	#Load data for Q1_B
	train_data = readFile('datasets/Q3_data.txt')
	X_train_np = train_data[:, :-1].astype('float')
	Y_train_np = train_data[:, -1].astype('str')

	#Scaling the data
	scaler = StandardScaling()
	X_train_np = scaler.fit_transform(X_train_np)

	label2cls = {'W': 0, 'M': 1}
	cls2label = {0: 'W', 1: 'M'}
	
	Y_cls_np = np.where(Y_train_np == 'W', 0, 1)
	#Homogenizing data by adding one's column to the data matrix.
	X_input = homogenize_data(X_train_np)
	#y_preds = []
	num_points, num_features = X_input.shape

	w = np.zeros(num_features)

	#Asserting that the logistic loss gradient computation is accurate
	loss, grad = logistic_loss(X_input, Y_cls_np, w)
	num_grad = numericalgrad(logistic_loss, X_input, Y_cls_np, w, 1e-5)
	assert np.linalg.norm(grad-num_grad)<1e-2, "Gradient computed is not matching the numerical gradient"

	#Standard Gradient Descent Function
	#standard gradient decent with learning rate of 0.01.
	w_learned = gradient_descent(X_input, Y_cls_np, logistic_loss, 1e-2, 1e-5, max_iterations=10000)

	#Gradient Descent with back tracking for faster model convergence and loss reduction
	#w_learned = gd_with_backtracking_v4(X_input, Y_cls_np, logistic_loss, 1, 1e-2, 1e-6)

	#Get Predictions
	y_preds = get_preds(X_input, w_learned)

	#Compute Accuracy
	trn_acc = cal_accuracy(Y_cls_np, y_preds)
	
	#Projecting the data into two dimensions
	
	PCA = pca_algo()
	data_projected = PCA.fit_transform(X_input, k=2)
	w_projected = PCA.transform(w_learned.reshape((1, -1)), k=2).reshape((-1))

	#making continuous grid for plotting
	xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = get_probs(grid, w_projected)[:, 1].reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
						vmin=0, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label("$P(y = 1)$")
	ax_c.set_ticks([0, .25, .5, .75, 1])

	ax.scatter(data_projected[:, 0], data_projected[:, 1], c=Y_cls_np, s=50,
			cmap="RdBu", vmin=-.2, vmax=1.2,
			edgecolor="white", linewidth=1)

	ax.set(aspect="equal",
		xlim=(-5, 5), ylim=(-5, 5),
		xlabel="$X_1$", ylabel="$X_2$")

	plt.title('Q3AB_Classification_Surface')
	plt.legend()
	plt.savefig("Q3AB_Visualization.jpg")
	
	print('END Q3_AB\n')


if __name__ == "__main__":
    main()
	