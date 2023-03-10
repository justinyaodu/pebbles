CXXFLAGS = -g -O3 -Wall -Wextra -Wshadow=local -march=native -std=c++17
SHARED_HEADERS = alloc.hpp bitset.hpp expr.hpp main.cpp spec.hpp synth.hpp timer.hpp util.hpp
FULL_TEST_HEADERS = alloc.hpp bitset.hpp expr.hpp test_sygus.cpp parser.cpp spec.hpp synth.hpp timer.hpp util.hpp
CPU_HEADERS = alloc_cpu.hpp
GPU_HEADERS = bitset_gpu.cu gpu_assert.cu

reference : reference.cpp parser.cpp alloc.hpp bitset.hpp expr.hpp spec.hpp synth.hpp timer.hpp util.hpp
	g++ $(CXXFLAGS) $^ -o $@

synth_cpu_st : synth_cpu_st.hpp $(SHARED_HEADERS) $(CPU_HEADERS)
	g++ -D SYNTH_VARIANT=1 $(CXXFLAGS) $^ -o $@

synth_cpu_mt : synth_cpu_mt.hpp $(SHARED_HEADERS) $(CPU_HEADERS)
	g++ -D SYNTH_VARIANT=2 -fopenmp $(CXXFLAGS) $^ -o $@

synth_gpu : main.cu synth_gpu.cu $(SHARED_HEADERS) $(GPU_HEADERS)
	nvcc -D SYNTH_VARIANT=3 -O3 -arch compute_61 --extended-lambda $< -o $@

synth_cpu_st_full_test : synth_cpu_st.hpp $(FULL_TEST_HEADERS) $(CPU_HEADERS)
	g++ -D SYNTH_VARIANT=1 $(CXXFLAGS) $^ -o $@

synth_cpu_mt_full_test : synth_cpu_mt.hpp $(FULL_TEST_HEADERS) $(CPU_HEADERS)
	g++ -D SYNTH_VARIANT=2 -fopenmp $(CXXFLAGS) $^ -o $@

synth_gpu_full_test : main.cu synth_gpu.cu $(FULL_TEST_HEADERS) $(GPU_HEADERS)
	nvcc -D SYNTH_VARIANT=3 -O3 -arch compute_61 --extended-lambda $< -o $@

gen_input : gen_input.cpp
	g++ $(CXXFLAGS) $^ -o $@

