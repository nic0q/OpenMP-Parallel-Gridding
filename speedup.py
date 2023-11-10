import numpy as np
from matplotlib import pyplot as plt

t_p = [5.3,
       5.44,5.55,5.77,5.35,5.48,5.45,5.53,5.55,5.59
       ]

t_s = 5.6
speedup = []
x = []
for i in range(0, len(t_p)):
    x = np.append(x, i+1)
    speedup = np.append(speedup, t_s/t_p[i])
    print(x[i])
plt.xlabel("N° de tareas")
plt.ylabel("Speedup")
plt.title("Comportamiento speedup / método matrices compartidas")
plt.plot(x, speedup)
plt.grid(color='lime', linestyle='--', linewidth=0.5)
plt.show()