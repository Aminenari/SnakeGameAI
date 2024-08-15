import numpy as np
import matplotlib.pyplot as plt
gamma_symbol = '\u03B3'
alpha_symbol = '\u03B1'


scores1 = np.load("15Alpha_0.01meanscore.npy")
scores2 = np.load("15Alpha_0.1meanscore.npy")
scores3 = np.load("15Alpha_0.3meanscore.npy")
scores4 = np.load("15Alpha_0.5meanscore.npy")
scores5 = np.load("15Alpha_0.7meanscore.npy")
scores6 = np.load("15Alpha_0.9meanscore.npy")

plt.figure()

plt.plot(scores1, label=f'{alpha_symbol} 0.01', color='blue')
plt.plot(scores2, label=f'{alpha_symbol} 0.1', color='red')
plt.plot(scores3, label=f'{alpha_symbol} 0.3', color='purple')
plt.plot(scores4, label=f'{alpha_symbol} 0.5', color='green')
plt.plot(scores5, label=f'{alpha_symbol} 0.7', color='black')
plt.plot(scores6, label=f'{alpha_symbol} 0.9', color='orange')




plt.xlabel('Games')
plt.ylabel('Mean score')
plt.legend()
plt.title('Title plot')
plt.grid(True)


plt.savefig('Nameplot.png')


plt.show()

