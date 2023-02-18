CXXFLAGS = -O3 -Wall -Wextra -Wshadow=local -march=native
SHARED_HEADERS = alloc.hpp bitset.hpp expr.hpp spec.hpp timer.hpp

reference : reference.cpp parser.cpp $(SHARED_HEADERS)
	g++ $(CXXFLAGS) $^ -o $@

synth_cpu_st : main.cpp synth_cpu_st.hpp synth_cpu.hpp $(SHARED_HEADERS)
	g++ -D SYNTH_VARIANT=1 $(CXXFLAGS) $^ -o $@

synth_cpu_mt : main.cpp synth_cpu_mt.hpp synth_cpu.hpp $(SHARED_HEADERS)
	g++ -D SYNTH_VARIANT=2 -fopenmp $(CXXFLAGS) $^ -o $@
