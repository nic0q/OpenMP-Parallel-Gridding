import numpy as np
from matplotlib import pyplot as plt

t_p = [6.64,2.76,2.13,1.72,1.63,1.62,1.61,1.69,1.47,1.58,1.56,1.65,1.774,1.52]
t_n = [1,2,3,5,6,7,8,9,10,15,20,30,50,100]

t_s = 5.6
speedup = []
x = []
for i in range(0, len(t_p)):
    x = np.append(x, t_n[i])
    speedup = np.append(speedup, t_s/t_p[i])
    print(x[i])
plt.xlabel("N° de tareas")
plt.ylabel("Speedup")
plt.title("Comportamiento speedup / Matrices Privadas")
plt.plot(x, speedup)
plt.grid(color='lime', linestyle='--', linewidth=0.5)
plt.show()