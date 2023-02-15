CXXFLAGS = -Wall -Wextra
SHARED_HEADERS = expr.hpp spec.hpp

reference : reference.cpp parser.cpp $(SHARED_HEADERS)
	g++ -O3 $(CXXFLAGS) $^ -o $@
