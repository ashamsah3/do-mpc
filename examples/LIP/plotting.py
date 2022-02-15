import numpy as np
import matplotlib.pyplot as plt
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


results = load_results('./results/LIP_turn.pkl')

x = results['mpc']['_x','x',0]
y = results['mpc']['_x','x',2]
xd = results['mpc']['_x','x',1]
yd = results['mpc']['_x','x',3]
px = results['mpc']['_u','u',0]
py = results['mpc']['_u','u',1]

xsim = results['simulator']['_x','x',0]
ysim = results['simulator']['_x','x',2]


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#ax2 = fig.add_subplot(3,1,2)
#ax3 = fig.add_subplot(3,1,3)
ax1.axis('equal')
ax1.plot(x,y)
ax1.scatter(-px+x,-py+y)
ax1.scatter(x,y)
circle1 = Circle((5, 8.5), 1, alpha=0.5)
circle2 = Circle((8, 9), 0.7, alpha=0.5)
goal = Circle((0.5, 9), 0.1, color="red", alpha=0.5)
ax1.add_artist(circle1)
ax1.add_artist(circle2)
ax1.add_artist(goal)
plt.title('LIP')
print(sqrt(px**2 + py**2))
#ax2.plot(x,xd)
#ax2.scatter(px+x,px*0)
#ax3.scatter(py,py*0)
#print(py)
#ax3.scatter(py+y,py*0)
plt.show()