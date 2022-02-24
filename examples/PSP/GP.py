import numpy as np
import scipy.stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from do_mpc.data import save_results, load_results






def GP(obs_x, obs_y, time, X, t0):
	y_train = obs_x
	y_train_y = obs_y

	X_train = time
	last =len(X)-1
	X = X[t0:last]





	kernel =  1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
	gaussian_process.fit(X_train, y_train)
	gaussian_process.kernel_
	mean_x, std_x = gaussian_process.predict(X.reshape(-1,1), return_std=True)


	


	kernel_y =  1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_process_y = GaussianProcessRegressor(kernel=kernel_y, n_restarts_optimizer=9)
	gaussian_process_y.fit(X_train, y_train_y)
	gaussian_process_y.kernel_
	mean_y, std_y = gaussian_process_y.predict(X.reshape(-1,1), return_std=True)

	return mean_x, std_x, mean_y, std_y



def pred(mean_x, std_x, mean_y, std_y, conf, horz):

	delta_x=sum(mean_x[0:horz])
	delta_y=sum(mean_y[0:horz])

	w=sum(std_x[0:horz]*conf)
	h=sum(std_y[0:horz]*conf)

	return delta_x, delta_y, h, w,












