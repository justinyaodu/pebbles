import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

with open('comparison_cvc4_pebbles.txt') as f:
    lines = f.readlines()
    indices = range(len(lines))
    cvc4_times = [float(line.split()[0]) for line in lines]
    pebbles_times = [float(line.split()[1]) for line in lines]

with open('multithread_2.txt') as f:
    lines = f.readlines()
    pebbles_times_mt = [float(line) for line in lines]

with open('gpu_version.txt') as f:
    lines = f.readlines()
    pebbles_times_gpu = [float(line) for line in lines if not line.startswith("Synthesizer")]

with open('found.txt') as f:
    lines = f.readlines()
    has_solution = ['T' in line for line in lines]

with open('singlethread10.txt') as f:
    lines = f.readlines()
    pebbles_times_st_random10 = [sum([float(line) for line in lines])] * 16

with open('multithread10.txt') as f:
    lines = f.readlines()
    pebbles_times_mt_random10 = [sum([float(line) for line in lines])] * 16

with open('gpu_version10.txt') as f:
    lines = f.readlines()
    pebbles_times_gpu_random10 = [sum([float(line) for line in lines if not line.startswith("Synthesizer")])] * 16

with open('singlethread100.txt') as f:
    lines = f.readlines()
    pebbles_times_st_random = [float(line) for line in lines]

with open('multithread100.txt') as f:
    lines = f.readlines()
    pebbles_times_mt_random = [float(line) for line in lines]

with open('gpu100.txt') as f:
    lines = f.readlines()
    pebbles_times_gpu_random = [float(line) for line in lines if not line.startswith("Synthesizer")]

with open('compareNumThreads10_2.txt') as f:
    lines = f.readlines()
    compare_thread_count_times = [float(line) for line in lines]
threadTimes = []
for i in range(16):
    threadTimes.append(sum(compare_thread_count_times[i*10:(i+1)*10]))

# Instead of directly sorting, get an array of what the indices should be to sort cvc4_times
# Then we sort both cvc4_times and pebbbles_times according to that order
sort_order = np.argsort(cvc4_times)
cvc4_times = np.array(cvc4_times)[sort_order]
pebbles_times = np.array(pebbles_times)[sort_order]
pebbles_times_mt = np.array(pebbles_times_mt)[sort_order]
pebbles_times_gpu = np.array(pebbles_times_gpu)[sort_order]

sort_order2 = np.argsort(pebbles_times_st_random)
pebbles_times_mt_random = np.array(pebbles_times_mt_random)[sort_order2]
pebbles_times_gpu_random = np.array(pebbles_times_gpu_random)[sort_order2]
pebbles_times_st_random = np.array(pebbles_times_st_random)[sort_order2]

cvc4_times_solved = cvc4_times[has_solution]
pebbles_times_solved = pebbles_times[has_solution]

cvc4_times_no_solution = cvc4_times[np.invert(has_solution)]
pebbles_times_no_solution = pebbles_times[np.invert(has_solution)]

plt.yscale('log')
plt.plot(cvc4_times_solved,'o',markersize=2, label="cvc4")
plt.plot(pebbles_times_solved,'o',markersize=2, label="pebbles")
plt.ylabel("Execution time in seconds")
plt.xlabel("SyGuS CrCi problems with solutions")
plt.title("Log-scale comparison of CVC4 1.8 and Pebbles")
plt.xticks([], []) # turn off x-ticks
leg = plt.legend(loc='upper left')
plt.savefig('cvc4_versus_pebbles_solved.png')
plt.show()

plt.yscale('log')
plt.plot(cvc4_times_no_solution,'o',markersize=2, label="cvc4")
plt.plot(pebbles_times_no_solution,'o',markersize=2, label="pebbles")
plt.ylabel("Execution time in seconds")
plt.xlabel("SyGuS CrCi problems with no solution")
plt.title("Log-scale comparison of CVC4 1.8 and Pebbles")
plt.xticks([], []) # turn off x-ticks
leg = plt.legend(loc='upper left')
plt.savefig('cvc4_versus_pebbles_unsolved.png')
plt.show()

plt.yscale('log')
plt.plot(pebbles_times,'o',markersize=2, label="singlethread")
plt.plot(pebbles_times_mt,'o',markersize=2, label="multithread")
plt.plot(pebbles_times_gpu,'o', markersize=2, label="gpu")
plt.ylabel("Execution time in seconds")
plt.xlabel("SyGuS CrCi problems")
plt.title("Comparison of Pebbles with single thread versus multithread")
plt.xticks([], []) # turn off x-ticks
leg = plt.legend(loc='upper left')
plt.savefig('single_versus_multi.png')
plt.show()

plt.yscale('log')
plt.plot(pebbles_times_st_random,'o', markersize=2, label="singlethread")
plt.plot(pebbles_times_mt_random,'o', markersize=2, label="multithread")
plt.plot(pebbles_times_gpu_random,'o', markersize=2, label="gpu")
plt.ylabel("Execution time in seconds")
plt.xlabel("Random 5 variable truth tables")
plt.title("Comparison of Pebbles with single thread versus multithread")
plt.xticks([], []) # turn off x-ticks
leg = plt.legend(loc='upper left')
plt.savefig('single_versus_multi_random100.png')
plt.show()

plt.yscale('log')
plt.plot(pebbles_times_st_random10,'o-',linewidth=1, markersize=2.5, label="singlethread")
plt.plot(threadTimes,'o-',linewidth=1, markersize=2.5, label="Multithread varying thread count")
plt.plot(pebbles_times_gpu_random10,'o-',linewidth=1, markersize=2.5, label="gpu")
#plt.plot(pebbles_times_mt_random10,'o-',linewidth=1, markersize=2.5, label="multithread")
plt.ylabel("Execution time in seconds (total all random benchmarks)")
plt.xlabel("Thread count")
plt.title("Comparison of multithread Pebbles with different thread counts")
plt.xticks(range(16), range(1,17)) # turn off x-ticks
leg = plt.legend(loc='upper left')
plt.savefig('thread_count_comparison.png')
plt.show()