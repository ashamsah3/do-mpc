from GP import *
from obstacle_traj import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import time





def CVaR_map(beta,horz, ind):


	dt =0.3
	#load obstacle trajectry
	xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(1,1, 8, 2, 5, 0.5)
	mem=3
	last=len(X)-1
	#GP 


	mean_x, std_x, mean_y, std_y= GP(d_obs_x[ind-mem:ind].reshape(-1,1), d_obs_y[ind-mem:ind].reshape(-1,1), X[ind-mem:ind], X, ind)
	ch = []
	conf = [.5, 1.96, 3, 5, 8]
	n=4
	m=5

	#fig2 = plt.figure()
	#ax1 = fig2.add_subplot(1,1,1)
	
	r = .2

	#predicition
	delta_x, delta_y, h, w = pred(mean_x, std_x, mean_y, std_y, 1, horz)
	#border = Rectangle((0,5), 8, 6, fc = "None", ec="black" )
	#ax.add_patch(border)
	#ax.axis('equal')

	v_cvar = []
	v_dist = []

	
	i=0
	x = np.arange(0, 8, 1)
	#print(x)
	y = np.arange(0, 6, 1)
	ii=0
	a=np.zeros((len(y),len(x)))
	for c in x:
		for r in y:
			#tic = time.perf_counter()
			var, cvar_x, dist_x = Var(1-beta,delta_x+xs[ind],w, c)
			var, cvar_y, dist_y = Var(1-beta,delta_y+ys[ind],h, r)
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



	#colormesh = ax1.pcolormesh(x,y,a, alpha=0.3)
	#fig2.colorbar(colormesh, ax=ax1)

	flat_a = a[::-1]
	flat_a = flat_a.flatten()
	#print(flat_a)
	#plt.show()

	return flat_a, a , x , y

