import numpy as np
import matplotlib.pyplot as plt

with open('comparison_cvc4_pebbles.txt') as f:
    lines = f.readlines()
    cvc4_times = [float(line.split()[0]) for line in lines]
    pebbles_times = [float(line.split()[1]) for line in lines]

# Instead of directly sorting, get an array of what the indices should be to sort cvc4_times
# Then we sort both cvc4_times and pebbbles_times according to that order
sort_order = np.argsort(cvc4_times)
cvc4_times = np.array(cvc4_times)[sort_order]
pebbles_times = np.array(pebbles_times)[sort_order]

plt.yscale('log')
plt.plot(cvc4_times,'ro--',linewidth=0.5, markersize=1.5, label="cvc4")
plt.plot(pebbles_times,'go--',linewidth=0.5, markersize=1.5, label="pebbles")
plt.ylabel("Execution time in seconds")
plt.xlabel("SyGuS CrCi problems")
plt.title("Log-scale comparison of CVC4 1.8 and Pebbles")
plt.xticks([], []) # turn off x-ticks
leg = plt.legend(loc='upper left')
plt.savefig('cvc4_versus_pebbles.png')
plt.show()
