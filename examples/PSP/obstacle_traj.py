import numpy as np
from do_mpc.data import save_results, load_results
from numpy.random import default_rng




def obstacle_obsrv(dt, x0, xf, y0, yf, inc):
	obstacle = load_results('./results/dubin.pkl')
	
	#xs0 = obstacle['mpc']['_x','x',0]
	#ys0 = obstacle['mpc']['_x','x',1]
	
	rng = np.random.default_rng()
	xs0 = (np.arange(x0, xf, inc)).reshape(-1,1)
	
	xs = xs0 #+ (0.01 * rng.standard_normal(size=len(xs0))).reshape(-1,1)
	#xs = xs0 #+ (np.random.uniform(-0.2,0.2,size=len(xs0))).reshape(-1,1)
	ys0 = xs*0 + y0 
	ys = ys0 + (0.05 * rng.standard_normal(size=len(xs0))).reshape(-1,1)
	#ys = ys0 #+ (np.random.uniform(-0.2,0.2,size=len(xs0))).reshape(-1,1)

	delta_xs = xs*0
	delta_ys = ys*0
	size=len(xs)
	X = np.arange(0, size*dt, dt).reshape(-1,1)
	for i in range(0,size-1,1):
		delta_xs[i]=xs[i+1]-xs[i]
		delta_ys[i]=ys[i+1]-ys[i]

	return xs, ys, delta_xs, delta_ys, X

