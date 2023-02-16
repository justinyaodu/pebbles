CXXFLAGS = -O3 -Wall -Wextra -march=native
SHARED_HEADERS = alloc.hpp bitset.hpp expr.hpp spec.hpp timer.hpp

reference : reference.cpp parser.cpp $(SHARED_HEADERS)
	g++ $(CXXFLAGS) $^ -o $@

synth_cpu_st : main.cpp synth_cpu_st.hpp $(SHARED_HEADERS)
	g++ $(CXXFLAGS) $^ -o $@
