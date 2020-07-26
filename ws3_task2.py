import numpy as np
def f(x,y):
    return x*x + 1.973*y*y
def theta_x(x):
    return 2*x
def theta_y(y):
    return 2*y*1.973
x0 = -200
y0 = -100
n=0.05
n_steps = 200

theta_vec = np.array([(x0), (y0)]) - n*np.array([theta_x(x0), theta_y(y0)])
for i in range(n_steps):
    theta_vec = theta_vec - n*np.array([theta_vec[0], theta_vec[1]])
    print('step: '+str(i) + ' | x0='+str(theta_vec[0]) +' | ' + str(theta_vec[0]))

