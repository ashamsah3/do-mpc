import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')
import do_mpc

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib.patches import Circle
from do_mpc.data import save_results, load_results


#results = load_results('./results/006_PSP_turn.pkl')
results = load_results('./results/dubin.pkl')

x = results['mpc']['_x','x',0]
y = results['mpc']['_x','x',1]
th = results['mpc']['_x','x',2]
v = results['mpc']['_u','u',0]
w = results['mpc']['_u','u',1]

dyn = results['mpc']['_tvp','dyn_obs',0]


fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.axis('equal')
ax1.plot(x,y)
ax1.scatter(x,y)
#ax1.scatter(((dyn-dyn)+1)*8,dyn)
circle1 = Circle((1.5, 9.25), 0.5, alpha=0.5)  
circle2 = Circle((3.5, 9.25), 0.5, alpha=0.5) # (2.5 , 9.25) >> turn_case_study, (2, 9.25) >>  001_turn_case_study
goal = Circle((4, 6), 0.1, color="red", alpha=0.5)
#ax1.add_artist(circle1)
#ax1.add_artist(circle2)
ax1.add_artist(goal)
plt.title('DMPC Dubin')
print(w)

#ax3.scatter(py+y,py*0)
plt.show()