import dubins
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from do_mpc.data import save_results, load_results
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle

q0 = (0, 0, 0)
q1 = (10, 2, 1)
turning_radius = 1.0
step_size = 0.05



X = np.arange(0, 20, 1).reshape(-1,1)

#results = load_results('./results/006_PSP_turn.pkl')
results = load_results('./results/001_dubin.pkl')

xs = results['mpc']['_x','x',0]
ys = results['mpc']['_x','x',1]
th = results['mpc']['_x','x',2]
us = results['mpc']['_u','u',0]
ws = results['mpc']['_u','u',1]

delta_xs = xs*0
delta_ys = ys*0
size=len(xs)
step_time=0.3
#print(len(xs))
X = np.arange(0, size*0.3, 0.3).reshape(-1,1)
#print(X)
for i in range(0,size-1,1):
	delta_xs[i]=xs[i+1]-xs[i]
	delta_ys[i]=ys[i+1]-ys[i]

#plt.plot(us*np.cos(th))
#plt.plot(delta_xs)
#plt.show()
prediction_history = delta_xs*0
std_history = delta_xs*0
last =len(xs)-1

mem =3
stop=8

for i in range(1, size-stop, 1):

	if i > mem:
		y_train = delta_xs[i-mem:i]
		X_train = X[i-mem:i]#xs[i-mem:i]
	else:
		y_train = delta_xs[0:i]
		X_train = X[0:i]#xs[0:i]
	kernel =  1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
	gaussian_process.fit(X_train, y_train)
	gaussian_process.kernel_
	mean_prediction, std_prediction = gaussian_process.predict(X.reshape(-1,1), return_std=True)
	#last_train =len(X_train)-1
	#prediction_h, std_h= gaussian_process.predict((X_train[last_train]+0.3).reshape(-1,1), return_std=True)
	#prediction_history[i-1]=prediction_h
	#std_history[i-1]=std_h


'''
	
	plt.plot(X, delta_xs, linestyle="dotted")
	plt.scatter(X_train, y_train, label="Obs")	
	plt.plot(X[i-mem:last], mean_prediction[i-mem:last], label="Mean prediction")
	plt.plot(X[i-mem:last], mean_prediction[i-mem:last].reshape(-1,1) + 1.96*std_prediction[i-mem:last].reshape(-1,1))
	plt.plot(X[i-mem:last], mean_prediction[i-mem:last].reshape(-1,1) - 1.96*std_prediction[i-mem:last].reshape(-1,1))
	plt.show()
'''	
	

for c in range(1, size-stop, 1):

	if c > mem:
		y_train_y = delta_ys[c-mem:c]
		X_train_y = X[c-mem:c]
	else:
		y_train_y = delta_ys[0:c]
		X_train_y = X[0:c]
	kernel_y =  1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_processy = GaussianProcessRegressor(kernel=kernel_y, n_restarts_optimizer=9)
	gaussian_processy.fit(X_train_y, y_train_y)
	gaussian_processy.kernel_
	mean_prediction_y, std_prediction_y = gaussian_processy.predict(X.reshape(-1,1), return_std=True)


	'''

	plt.plot(ys, delta_ys, linestyle="dotted")
	plt.scatter(X_train_y, y_train_y, label="Obs")	
	plt.plot(ys[c-mem:last], mean_prediction_y[c-mem:last], label="Mean prediction")
	plt.plot(ys[c-mem:last], mean_prediction_y[c-mem:last].reshape(-1,1) + 3*std_prediction_y[c-mem:last].reshape(-1,1))
	plt.plot(ys[c-mem:last], mean_prediction_y[c-mem:last].reshape(-1,1) - 3*std_prediction_y[c-mem:last].reshape(-1,1))
	
	
	plt.show()
	'''


#std_history=std_history.reshape((20,))
#prediction_history=prediction_history.reshape((20,))

#plt.errorbar(X,prediction_history, yerr=std_history*2, barsabove=True, label="prediction")
#plt.scatter(X,delta_xs, label="Observations")
#plt.plot(X,delta_xs-(prediction_history).reshape(-1,1))
#plt.legend()




fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)


ax1.title.set_text('Euclidian Space')
ax2.title.set_text('Delta x VS time')
ax3.title.set_text('Delta y VS time')

ax1.scatter(xs,ys)
ax2.plot(X,us*np.cos(th))
ax3.plot(X,us*np.sin(th))

dt=0.3
n_horz = 1
#print(X[i]+dt)
#print(mean_prediction)

pred_t = X[i-1]+ dt*n_horz

pred_vx= gaussian_process.predict((pred_t).reshape(-1,1))
pred_x=xs[i-1]+ pred_vx
pred_vy= gaussian_processy.predict((pred_t).reshape(-1,1))
pred_y=ys[c-1]+ pred_vy
#print(pred_vx)

ax1.scatter(xs[i-1], ys[c-1], c="orange", label="last obs")
ax1.scatter(pred_x, pred_y, c="red", label="mean prediction")

ax2.plot(X[i-mem:last], mean_prediction[i-mem:last], label="Mean prediction")

ax2.fill_between(
	    X[i-mem:last].ravel(),
	    (mean_prediction[i-mem:last] - 1.96*std_prediction[i-mem:last].reshape(-1,1)).ravel(),
	    (mean_prediction[i-mem:last] + 1.96*std_prediction[i-mem:last].reshape(-1,1)).ravel(),
	    alpha=0.25,
	    label=r"95% confidence interval",
	)
#ax2.plot(X[i-mem:last], mean_prediction[i-mem:last] + 2*std_prediction[i-mem:last].reshape(-1,1), label="2 std", color="red")
#ax2.plot(X[i-mem:last], mean_prediction[i-mem:last] - 2*std_prediction[i-mem:last].reshape(-1,1), label="2 std", color="red")
ax2.scatter(X,us*np.cos(th))
ax2.scatter(X_train, y_train, label="Obs")	
ax2.scatter(pred_t, pred_vx, label="pred")
ax2.legend()	

ax3.plot(X[c-mem:last], mean_prediction_y[c-mem:last], label="Mean prediction")
ax3.fill_between(
	    X[i-mem:last].ravel(),
	    (mean_prediction_y[i-mem:last] - 1.96*std_prediction_y[i-mem:last].reshape(-1,1)).ravel(),
	    (mean_prediction_y[i-mem:last] + 1.96*std_prediction_y[i-mem:last].reshape(-1,1)).ravel(),
	    alpha=0.25,
	    label=r"95% confidence interval",
	)
ax3.scatter(X,us*np.sin(th))
ax3.scatter(X_train_y, y_train_y, label="Obs")	
ax3.scatter(pred_t, pred_vy, label="pred")
ax3.legend()


sum_til = 5
sum_deltax=sum(mean_prediction[i:i+sum_til])
sum_deltax_up = sum((mean_prediction[i:i+sum_til] + 1.96*std_prediction[i:i+sum_til].reshape(-1,1)).ravel())
sum_deltax_dw = sum((mean_prediction[i:i+sum_til] - 1.96*std_prediction[i:i+sum_til].reshape(-1,1)).ravel())

sum_deltay=sum(mean_prediction_y[i:i+sum_til])
sum_deltay_up = sum((mean_prediction_y[i:i+sum_til] + 1.96*std_prediction_y[i:i+sum_til].reshape(-1,1)).ravel())
sum_deltay_dw = sum((mean_prediction_y[i:i+sum_til] - 1.96*std_prediction_y[i:i+sum_til].reshape(-1,1)).ravel())

xworst_u=[]
yworst_u=[]
xworst_d=[]
yworst_d=[]

conf=0.5
for b in range(i,i+sum_til+1,1):
	xworst_u.append(sum((mean_prediction[i:b] + conf*std_prediction[i:b].reshape(-1,1)).ravel()))
	yworst_u.append(sum((mean_prediction_y[i:b] + conf*std_prediction_y[i:b].reshape(-1,1)).ravel()))
	xworst_d.append(sum((mean_prediction[i:b] - conf*std_prediction[i:b].reshape(-1,1)).ravel()))
	yworst_d.append(sum((mean_prediction_y[i:b] - conf*std_prediction_y[i:b].reshape(-1,1)).ravel()))
	

ax1.plot(xs[i-1]+xworst_u,ys[c-1]+yworst_u, c="black")
ax1.plot(xs[i-1]+xworst_u,ys[c-1]+yworst_d, c="black")
ax1.plot(xs[i-1]+xworst_d,ys[c-1]+yworst_u, c="black")
ax1.plot(xs[i-1]+xworst_d,ys[c-1]+yworst_d, c="black")



ax1.scatter(xs[i-1]+ sum_deltax, ys[c-1]+sum_deltay, c="black", label="obs")
#ax1.scatter(xs[i-1]+ sum_deltax_up, ys[c-1]+sum_deltay_up, label="++2std")
#ax1.scatter(xs[i-1]+ sum_deltax_up, ys[c-1]+sum_deltay_dw, label="+-2std")
#ax1.scatter(xs[i-1]+ sum_deltax_dw, ys[c-1]+sum_deltay_up, label="-+2std")
#ax1.scatter(xs[i-1]+ sum_deltax_dw, ys[c-1]+sum_deltay_dw, label="--2std")

ellipse2 = Ellipse(xy=(xs[i-1]+ sum_deltax, ys[c-1]+sum_deltay), width=2*sum((conf*std_prediction[i:b].reshape(-1,1)).ravel()),
					    height=2*sum((conf*std_prediction_y[i:b].reshape(-1,1)).ravel()), 
                        edgecolor='black', fc='None', lw=2)
ax1.add_patch(ellipse2)

print(sum((conf*std_prediction[i:b].reshape(-1,1)).ravel()))
print(sum((conf*std_prediction_y[i:b].reshape(-1,1)).ravel()))

meanx, stdx = gaussian_process.predict((pred_t).reshape(-1,1), return_std=True)
meany, stdy = gaussian_processy.predict((pred_t).reshape(-1,1), return_std=True)
#mean, covx = gaussian_process.predict((pred_t).reshape(-1,1), return_cov=True)
#mean, covy = gaussian_processy.predict((pred_t).reshape(-1,1), return_cov=True)

ax1.axis('equal')
#circle1 = Circle((pred_x, pred_y), 1.96*stdx, fc="None", ec="red")  
#circle2 = Circle((pred_x, pred_y), 1.96*stdy, fc="None", ec="blue")  
#ax1.add_artist(circle1)
#ax1.add_artist(circle2)
ellipse1 = Ellipse(xy=(pred_x, pred_y), width=2*stdx*1.96, height=2*stdy*1.96, 
                        edgecolor='r', fc='None', lw=2)
#ax1.add_patch(ellipse1)



midpoint_x = (sum_deltax)/2 #+ xs[i-1]
midpoint_y = (sum_deltay)/2 #+ ys[i-1]

a = (midpoint_x - (sum_deltax)) + std_prediction[i+sum_til]*1.96
b = (midpoint_y - (sum_deltay)) + std_prediction_y[i+sum_til]*1.96

#ax1.scatter(midpoint_x+ xs[i-1], midpoint_y + ys[i-1], c="y", label="C")

ellipse3 = Ellipse(xy=(midpoint_x+ xs[i-1], midpoint_y + ys[i-1]), width=2*a, height=2*b, 
                        edgecolor='y', fc='None', lw=2)
#ax1.add_patch(ellipse3)


'''
print(meanx)
print(stdx)
Z = ((meanx+0.1)-meanx)/stdx 
print(Z)
p = scipy.stats.norm.sf(abs(Z))
print(p)
'''


'''
std_x1= xs[i]+ n_horz*dt*(pred_vx+(1.96*stdx))
std_x2= xs[i]+ n_horz*dt*(pred_vx-(1.96*stdx))
std_y1= ys[i]+ n_horz*dt*(pred_vy+(1.96*stdy))
std_y2= ys[i]+ n_horz*dt*(pred_vy-(1.96*stdy))
ax1.scatter(std_x1, pred_y, c="red", label="mean prediction")
ax1.scatter(std_x2, pred_y, c="red", label="mean prediction")
ax1.scatter(pred_x, std_y1, c="red", label="mean prediction")
ax1.scatter(pred_x, std_y2, c="red", label="mean prediction")


ellipse = Ellipse(xy=(pred_x, pred_y), width=std_x/2, height=std_y/2, 
                        edgecolor='r', fc='None', lw=2)
ax1.add_patch(ellipse)
'''
ax1.legend()	
plt.show()



