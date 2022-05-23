import numpy as np
from discret_risk_map import *
from eucl2disc import *
from GridWorld import GridWorld
from GP import *
from obstacle_traj import *
from ValueIteration import ValueIteration
from scipy.interpolate import UnivariateSpline
import time


xfull=[]
yfull=[]
x=0.5
y=0.5
xfull.append(x)
yfull.append(y)

xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(1,1, 8, 2, 5, 0.5)
xs2, ys2, d_obs_x2, d_obs_y2, X2 = obstacle_obsrv(1,8, 1, 2, 5, -0.5)

problem = GridWorld('data/world004.csv', reward={0: 0.0, 1: 5.0, 2: -5.0, 3: np.NaN}, random_rate=0.2)
mem = 3
beta =0.95
x_space = np.arange(0, 8, 1)
y_space = np.arange(0, 6, 1)
for i in range(3,14):
	
	mean_x, std_x, mean_y, std_y = GP(d_obs_x[i-mem:i].reshape(-1,1), d_obs_y[i-mem:i].reshape(-1,1), X[i-mem:i], X, i)
	conf = 1.96
	horz = 1
	delta_x, delta_y, h, w = pred(mean_x, std_x, mean_y, std_y, conf, horz)

	mean_x2, std_x2, mean_y2, std_y2 = GP(d_obs_x2[i-mem:i].reshape(-1,1), d_obs_y2[i-mem:i].reshape(-1,1), X2[i-mem:i], X2, i)
	conf = 1.96
	horz = 1
	delta_x2, delta_y2, h2, w2 = pred(mean_x2, std_x2, mean_y2, std_y2, conf, horz)
	
	ind = int(i)
	#CVaR, CVaR_space , x_space, y_space = CVaR_map(0.95, 1, ind)

	CVaR1, CVaR_space1 = CVaR_map_mpc(beta, delta_x, delta_y, w, h, 8, 6, xs[i], ys[i])
	CVaR2, CVaR_space2 = CVaR_map_mpc(beta, delta_x2, delta_y2, w2, h2, 8, 6, xs2[i], ys2[i])

	CVaR = CVaR2 + CVaR1
	CVaR_space = CVaR_space1+ CVaR_space2
	#tic = time.perf_counter()
	solver = ValueIteration(problem.reward_function, problem.transition_model, CVaR, gamma=0.99)
	solver.train()
	#problem.visualize_value_policy(policy=solver.policy, values=solver.values)

	s = eucl2disc(x,y,8,6,1)
	g = 23
	#toc = time.perf_counter()
	#print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
	
	while s != g:
		action = solver.policy[s]
		#symbol = ['^', '>', 'v', '<']
		if action == 0:
			yfull.append(y+1)
			xfull.append(x)
			y=y+1
		elif action == 1:
			xfull.append(x+1)
			yfull.append(y)
			x=x+1
		elif action == 2:
			yfull.append(y-1)
			xfull.append(x)
			y=y-1
		elif action == 3:
			xfull.append(x-1)
			yfull.append(y)
			x=x-1
		s = eucl2disc(x,y,8,6,1)
	
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(1,1,1)
	colormesh = ax1.pcolormesh(x_space+0.5,y_space+0.5,CVaR_space, alpha=0.3)
	#fig2.colorbar(colormesh, ax=ax1)
	ax1.scatter(xfull,yfull)
	border = Rectangle((0,0), 8, 6, fc = "None", ec="black" )
	ax1.add_patch(border)
	wall = Rectangle((3,0), 1, 1, fc = "gray", ec="black" )
	wall2 = Rectangle((3,3), 1, 3, fc = "gray", ec="black" )
	goal = Rectangle((7,3), 1, 1, fc = "green", ec="black" )
	ax1.add_patch(wall)
	ax1.add_patch(wall2)
	ax1.add_patch(goal)
	#xs = np.linspace(xfull[0], 8, 1000)
	#path = UnivariateSpline(xfull, yfull)
	#ax1.scatter(xfull,path(xfull))
	plt.show()

	x = xfull[1]
	y=yfull[1]
	xfull=[]
	yfull=[]
	xfull.append(x)
	yfull.append(y)
	
	

#problem.random_start_policy(policy=solver.policy, start_pos=(1, 1), n=1000)

#problem.visualize_value_policy(policy=solver.policy, values=solver.values)
'''
xfull=[]
yfull=[]
x=1.5
y=2.5
xfull.append(x)
yfull.append(y)
s = eucl2disc(x,y,8,6,1)
g = 23

while s != g:
	action = solver.policy[s]
	#symbol = ['^', '>', 'v', '<']
	if action == 0:
		yfull.append(y+1)
		xfull.append(x)
		y=y+1
	elif action == 1:
		xfull.append(x+1)
		yfull.append(y)
		x=x+1
	elif action == 2:
		yfull.append(y-1)
		xfull.append(x)
		y=y-1
	elif action == 3:
		xfull.append(x-1)
		yfull.append(y)
		x=x-1
	s = eucl2disc(x,y,8,6,1)
	#print(s)

'''
#fig2 = plt.figure()
#ax1 = fig2.add_subplot(2,1,1)
#ax1.scatter(xfull,yfull)
plt.show()

