import numpy as np
import matplotlib.pyplot as plt

with open('comparison_cvc4_pebbles.txt') as f:
    lines = f.readlines()
    cvc4_times = [float(line.split()[0]) for line in lines]
    pebbles_times = [float(line.split()[1]) for line in lines]

#cvc4_times = [min(5,y) for y in cvc4_times]

print(cvc4_times)
#print(pebbles_times)

plt.yscale('log')
plt.plot(range(len(cvc4_times)),cvc4_times,'ro--',linewidth=0.5, markersize=1.5, label="cvc4")
plt.plot(range(len(cvc4_times)),pebbles_times,'go--',linewidth=0.5, markersize=1.5, label="pebbles")
leg = plt.legend(loc='upper left')
plt.savefig('cvc4_versus_pebbles2.png')
plt.show()
