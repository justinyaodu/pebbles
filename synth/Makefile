CXXFLAGS = -g -O3 -Wall -Wextra -Wshadow=local -march=native
SHARED_HEADERS = alloc.hpp bitset.hpp expr.hpp main.cpp spec.hpp synth.hpp timer.hpp util.hpp
CPU_HEADERS = alloc_cpu.hpp
GPU_HEADERS = bitset_gpu.cu gpu_assert.cu

reference : reference.cpp parser.cpp alloc.hpp bitset.hpp expr.hpp spec.hpp synth.hpp timer.hpp util.hpp
	g++ $(CXXFLAGS) $^ -o $@

synth_cpu_st : synth_cpu_st.hpp $(SHARED_HEADERS) $(CPU_HEADERS)
	g++ -D SYNTH_VARIANT=1 $(CXXFLAGS) $^ -o $@

synth_cpu_mt : synth_cpu_mt.hpp $(SHARED_HEADERS) $(CPU_HEADERS)
	g++ -D SYNTH_VARIANT=2 -fopenmp $(CXXFLAGS) $^ -o $@

synth_gpu : main.cu synth_gpu.cu $(SHARED_HEADERS) $(GPU_HEADERS)
	nvcc -D SYNTH_VARIANT=3 -O3 -arch compute_61 $< -o $@
