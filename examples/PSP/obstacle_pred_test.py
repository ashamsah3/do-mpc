from GP import *
from obstacle_traj import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


dt =0.3
#load obstacle trajectr
xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(1,2, -2, 2, 0, -0.2)




#t0 = X[0]
tk = 10
mem=3
last=len(X)-1

mean_x, std_x, mean_y, std_y= GP(d_obs_x[tk-mem:tk].reshape(-1,1), d_obs_y[tk-mem:tk].reshape(-1,1), X[tk-mem:tk], X, tk)


conf = 1.96
horz = 1
delta_x, delta_y, h, w = pred(mean_x, std_x, mean_y, std_y, conf, horz)

print(h)
print(w)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax2.plot(X,d_obs_x)
ax2.scatter(X[tk],d_obs_x[tk])
ax2.plot(X[tk:last],mean_x)
ax1.scatter(xs,ys)
ax1.scatter(xs[tk],ys[tk])
ax1.scatter(xs[tk]+delta_x,ys[tk]+delta_y)


ellipse2 = Ellipse(xy=(xs[tk]+delta_x, ys[tk]+delta_y), width=2*w,
					    height=2*h, 
                        edgecolor='black', fc='None', lw=2)
ax1.add_patch(ellipse2)


beta = 0.95
var, cvar = Var(1-beta,delta_x,w)
print(var)
print(cvar)

plt.show()
