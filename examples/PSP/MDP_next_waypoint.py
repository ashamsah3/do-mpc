import numpy as np
from discret_risk_map import *
from eucl2disc import *
from GridWorld import GridWorld
from ValueIteration import ValueIteration


def next_waypoint(s, CVaR, dx, lenx, leny):


	problem = GridWorld('data/no_walls.csv', reward={0: -1, 1: 5.0, 2: -0.1, 3: np.NaN}, random_rate=0.2)


	#tic = time.perf_counter()
	solver = ValueIteration(problem.reward_function, problem.transition_model, CVaR, gamma=0.99)
	solver.train()
	#problem.visualize_value_policy(policy=solver.policy, values=solver.values)

	#s = eucl2disc(x,y,lenx,leny,dx)
	
	
	
	action = solver.policy[s]
	#symbol = ['^', '>', 'v', '<']
	if action == 0:
		wy = 1
		wx = 0
		s = s - lenx
	elif action == 1:
		wx = 1
		wy = 0
		s = s + 1
	elif action == 2:
		wy = - 1
		wx = 0
		s = s - lenx
	elif action == 3:
		wx = -1
		wy = 0
		s = s - 1



	return wx, wy, s
	
