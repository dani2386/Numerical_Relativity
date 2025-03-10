import numpy as np
import matplotlib.pyplot as plt
from src.WEConvergengeTester import WEConvergenceTester


X = 100
delta_x = 1
courant_param = 1
T = 20

init = [lambda x: np.exp(-pow(x - 20, 2)/(2 * pow(2 , 2))),
        lambda x: (x - 20) / pow(2, 2) * np.exp(-pow(x - 20, 2)/(2 * pow(2 , 2)))]

f = lambda x, t: np.exp(-pow(x - t - 20, 2) / 4)

tester = WEConvergenceTester(X, init, T, f)

plt.plot(tester.norm_convergence(courant_param, delta_x, 6))
plt.show()
