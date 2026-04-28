#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

# Number of sample points for plotting.
n = 50

def novelty(p2, p6, eps = 1e-8):
    relevance = p2 + p6
    pt2 = p2 / (relevance + eps)
    pt6 = p6 / (relevance + eps)
    h_rel = -(p2 * np.log2(pt2) + p6 * np.log2(pt6))
    return np.clip(h_rel, 0, 1)


#rng = np.random.default_rng(seed = 42)

#points = rng.uniform(size = (2, n)
# Rotate impossible points 180 degrees around (0.5, 0.5).
# This happens to be equivalent to (x,y) => (1-x, 1-y)
#impossible = points.sum(axis = 0) > 1
#points[:,impossible] = 1 - points[:,impossible]

# Alright, now points should be uniformly distributed
# over the valid triangle.

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
points = np.meshgrid(x, y)
valid = points[0] + points[1] <= 1
valid_x = points[0][valid]
valid_y = points[1][valid]

# Also construct a plane to differentiate valid from invalid.
toggle = np.arange(2)
x_pl, z_pl = np.meshgrid(toggle, toggle)
y_pl = 1 - x_pl

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.plot_surface(points[0], points[1], novelty(*points))
ax.plot_surface(x_pl, y_pl, z_pl, color = (0, 0, 0, 0.5), label = 'Validity boundary')
ax.plot_trisurf(valid_x, valid_y, novelty(valid_x, valid_y), color = '#1f77b4D0', label = 'Novelty')

ax.legend()
ax.set_xlabel(r'$\Pr[\textrm{digit}=2]$')
ax.set_ylabel(r'$\Pr[\textrm{digit}=6]$')
ax.set_zlabel('Novelty Score')

ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])

ax.view_init(35, -80, 0)

plt.savefig('novelty.png', dpi=600)
plt.show()
