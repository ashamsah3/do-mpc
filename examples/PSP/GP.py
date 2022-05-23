import numpy as np
import scipy.stats
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from do_mpc.data import save_results, load_results
import time





def GP(obs_x, obs_y, time, X, t0):
	y_train = obs_x.reshape(-1,1)
	y_train_y = obs_y.reshape(-1,1)

	X_train = time.reshape(-1,1)
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


def Var(q, mean, std, x):
	dist = mean-x
	var = norm.ppf(q,dist,std)
	#tic = time.perf_counter()
	cvar = (1/(1-q))*scipy.stats.norm.expect(lambda x: x, lb = var)
	#toc = time.perf_counter()
	#print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")


	return var, cvar, dist

def CVaR_map_mpc(beta, delta_x, delta_y, w, h, x, y, ox_c, oy_c):
	i=0
	x_space = np.arange(0, x, 1)
	y_space = np.arange(0, y, 1)
	ii=0
	a=np.zeros((len(y_space),len(x_space)))
	for c in x_space:
		for r in y_space:
			#tic = time.perf_counter()
			var, cvar_x, dist_x = Var(1-beta,delta_x+ox_c,w, c)
			var, cvar_y, dist_y = Var(1-beta,delta_y+oy_c,h, r)
			#toc = time.perf_counter()
			#print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
			#v_cvar.append(cvar)
			#v_dist.append(dist)
			#print(i)
			#print(ii)
			a[ii][i]=cvar_x+cvar_y
			ii=ii+1
		i=i+1
		ii=0

	flat_a = a[::-1]
	flat_a = flat_a.flatten()

	return flat_a, a










