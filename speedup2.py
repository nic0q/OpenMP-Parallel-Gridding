import numpy as np
from matplotlib import pyplot as plt

t_p = [10.326,3.95,2.458,2.168,2.102,1.792,1.78,1.624,1.63,1.486,1.628,1.562]
t_n = [1,2,3,4,5,6,7,8,9,10,15,20]

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