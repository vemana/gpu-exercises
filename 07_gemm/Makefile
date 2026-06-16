NVCC = nvcc
NVCC_FLAGS = -O3 -arch=native -lcublas
TARGET = run_test

SRCS = main.cc kernel.cu reference_kernel.cu
HEADERS = kernel.h reference_kernel.h ../utils/framework.h ../utils/utils.h ../utils/tracer.h ../utils/argparse.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
