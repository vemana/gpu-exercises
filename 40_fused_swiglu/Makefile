NVCC = nvcc
CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17
NVCCFLAGS = -O3 -arch=native -std=c++17

TARGET = bin/run_test

all: $(TARGET)

$(TARGET): main.cc kernel.cu reference_kernel.cu
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $^ -o $@
	cp ../utils/run_test.sh bin/
	cp ../utils/run_profiler.sh bin/
	chmod +x bin/run_test.sh bin/run_profiler.sh

clean:
	rm -rf bin
