import numpy as np

def eucl2disc(x,y,lenx,leny,dx):
	sx=np.floor(x)/dx + 1
	sy=np.floor(y)/dx + 1
	n = sx
	m=leny-sy
	s = -1 + n + lenx*m 
	s = int(s)
	return s 



