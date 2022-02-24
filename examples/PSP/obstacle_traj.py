import numpy as np
from do_mpc.data import save_results, load_results




def obstalce_obsrv(dt):
	obstacle = load_results('./results/dubin.pkl')

	xs = obstacle['mpc']['_x','x',0]
	ys = obstacle['mpc']['_x','x',1]
	delta_xs = xs*0
	delta_ys = ys*0
	size=len(xs)
	X = np.arange(0, size*dt, dt).reshape(-1,1)
	for i in range(0,size-1,1):
		delta_xs[i]=xs[i+1]-xs[i]
		delta_ys[i]=ys[i+1]-ys[i]

	return xs, ys, delta_xs, delta_ys, X

