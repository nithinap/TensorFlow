import matplotlib
import numpy as np
matplotlib.use('Agg')

import matplotlib.pyplot as plt

n = 50
x = np.random.rand(n)
y = np.random.rand(n)
area = np.pi * (10 * np.random.rand(n))**2
colors = np.random.rand(n)

print(colors)

plt.scatter(x, y, s=area, c=colors, alpha=0.5) # 0 to 10 point radiuses
plt.savefig('scatter-plot.png')