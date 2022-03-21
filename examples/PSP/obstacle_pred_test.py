from GP import *
from obstacle_traj import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


dt =0.3
#load obstacle trajectr
xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(1,2, -2, 2, 0, -0.2)

#t0 = X[0]
tk = 8
mem=3
last=len(X)-1

mean_x, std_x, mean_y, std_y= GP(d_obs_x[tk-mem:tk].reshape(-1,1), d_obs_y[tk-mem:tk].reshape(-1,1), X[tk-mem:tk], X, tk)

ch = []
conf = [.5, 1.96, 3, 5, 8]
n=8
m=5

for j in range(m):
  for i in range(n):
    horz = i+1
    delta_x, delta_y, h, w = pred(mean_x, std_x, mean_y, std_y, conf[1], horz)
    ch.append((delta_x, delta_y, h, w))

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)
# ax2.plot(X,d_obs_x)
# ax2.scatter(X[tk],d_obs_x[tk])
# ax2.plot(X[tk:last],mean_x)

#for framing
ax1.margins(0.12,0.12)

r = .2

colormap = plt.get_cmap('summer')

# ax1.scatter(xs[0:tk],ys[0:tk])

tk_b4 = 3
for i in range(tk_b4):
  #print(colormap(0, alpha=i/tk_b4))
  e = Ellipse(xy=(xs[tk-i-1], ys[tk-i-1]), width=2*r, height=2*r, facecolor=colormap(0), alpha=(tk_b4 - i + 1)/(tk_b4 + 2), lw=2)
  ax1.add_patch(e)

ax1.add_patch(Ellipse(xy=(xs[tk], ys[tk]), width=2*r, height=2*r, facecolor='orange', lw=2))

# remove alpha on these to see the points. they are just for framing the window not meant to be drwn
ax1.scatter(xs[tk-tk_b4]-r,ys[tk-tk_b4]+r, alpha=0)
ax1.scatter(xs[tk]+delta_x+(ch[-1][2]+r),ys[tk]+delta_y-(ch[-1][3]+r), alpha=0)

for i in range(len(ch)):
  ellipse2 = Ellipse(xy=(xs[tk]+ch[i][0], ys[tk]+ch[i][1]), width=ch[i][2]+2*r,
                height=ch[i][3]+2*r, edgecolor=colormap((m - i//n)/m), fc='None', lw=2)
  ax1.add_patch(ellipse2)

delta_x, delta_y, h, w = pred(mean_x, std_x, mean_y, std_y, conf[1], 4)
beta = 0.95
#print(delta_x)
#print(w)
x = np.arange(-7, 14, 0.1)
v_cvar = []
v_dist = []
for i in x:
	var, cvar, dist = Var(1-beta,delta_x+xs[tk],w, i)
	v_cvar.append(cvar)
	v_dist.append(dist)

ax1.plot(x,v_cvar)
ax1.axis('equal')

plt.show()