NVCC = nvcc
NVCC_FLAGS = -O3 -arch=native

TARGET = bin/run_test
SRCS = main.cc kernel.cu reference_kernel.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)
	@cp ../utils/run_test.sh bin/
	@cp ../utils/run_profiler.sh bin/
	@chmod +x bin/run_test.sh bin/run_profiler.sh

clean:
	rm -rf bin
