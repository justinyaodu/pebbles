import numpy as np
import matplotlib.pyplot as plt

with open('comparison_cvc4_pebbles.txt') as f:
    lines = f.readlines()
    cvc4_times = [line.split()[0] for line in lines]
    pebbles_times = [line.split()[1] for line in lines]

plt.plot(cvc4_times)
plt.plot(pebbles_times)
plt.show()
plt.savefig('cvc4_versus_pebbles.png')