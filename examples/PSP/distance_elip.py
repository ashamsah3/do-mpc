import math
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches
from matplotlib.patches import Ellipse

plot_size = (640, 480)
plots = [
    (240, 120),
    (240, 200)]

def solve(semi_major, semi_minor, p):
    px = abs(p[0])
    py = abs(p[1])

    t = math.pi / 4

    a = semi_major
    b = semi_minor

    for x in range(0, 3):
        x = a * math.cos(t)
        y = b * math.sin(t)

        ex = (a*a - b*b) * math.cos(t)**3 / a
        ey = (b*b - a*a) * math.sin(t)**3 / b

        rx = x - ex
        ry = y - ey

        qx = px - ex
        qy = py - ey

        r = math.hypot(ry, rx)
        q = math.hypot(qy, qx)

        delta_c = r * math.asin((rx*qy - ry*qx)/(r*q))
        delta_t = delta_c / math.sqrt(a*a + b*b - x*x - y*y)

        t += delta_t
        t = min(math.pi/2, max(0, t))

    return (math.copysign(x, p[0]), math.copysign(y, p[1]))

a = 5
b= 8
px =10 * np.cos(math.pi /4)
py = 10 * np.sin(math.pi /4)
p = solve(a,b,(px,py))
d = math.hypot(py-p[1], px-p[0])
ellipse1 = Ellipse(xy=(0,0), width=2*a, height=2*b, edgecolor='r', fc='None', lw=2)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.add_patch(ellipse1)
ax1.scatter(px,py)
ax1.axis('equal')
print(d)
print(np.sqrt((px)**2+(py**2))-(5))
plt.show()

