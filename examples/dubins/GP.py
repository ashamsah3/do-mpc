import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


X = np.linspace(start=0, stop=10, num=1_0).reshape(-1,1)
y =np.squeeze(X*0)

plt.plot(X, y, linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")



rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
#X_train, y_train = X[training_indices], y[training_indices]

for i in range(1, 10, 1):
	X_train, y_train = X[0:i], y[0:i]
	kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
	gaussian_process.fit(X_train, y_train)
	gaussian_process.kernel_

	mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

	plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
	plt.scatter(X_train, y_train, label="Observations")
	plt.plot(X, mean_prediction, label="Mean prediction")
	plt.fill_between(
	    X.ravel(),
	    mean_prediction - 1.96* std_prediction,
	    mean_prediction + 1.96* std_prediction,
	    alpha=0.5,
	    label=r"95% confidence interval",
	)

	plt.legend()
	plt.xlabel("$x$")
	plt.ylabel("$f(x)$")
	_ = plt.title("Gaussian process regression on noise-free dataset")
	plt.show()

print(std_prediction.shape)
#ysample=gaussian_process.sample_y([[6]],n_samples=10, random_state=0)
#print(ysample)
#plt.scatter(6+(ysample*0), ysample)

