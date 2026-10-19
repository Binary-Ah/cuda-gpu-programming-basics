NVCC ?= nvcc
ARCH ?= sm_86
NVFLAGS = -O2 -arch=$(ARCH)
BIN = bin

all: $(BIN)/hello $(BIN)/vector_add_cpu $(BIN)/vector_add_gpu $(BIN)/pinned_memory $(BIN)/sgemm $(BIN)/conv2d

$(BIN):
	mkdir -p $(BIN)
$(BIN)/hello: 01_hello/hello.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<
$(BIN)/vector_add_cpu: 02_vector_add_cpu_vs_gpu/cpu.c | $(BIN)
	gcc -O0 -I02_vector_add_cpu_vs_gpu -o $@ $<
$(BIN)/vector_add_gpu: 02_vector_add_cpu_vs_gpu/gpu.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -I02_vector_add_cpu_vs_gpu -o $@ $<
$(BIN)/pinned_memory: 03_pinned_memory/pinned_memory.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<
$(BIN)/sgemm: 04_cublas_sgemm/gemm.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $< -lcublas
$(BIN)/conv2d: 05_conv2d_im2col_cublas/conv2d_cublas.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $< -lcublas

run: all
	mkdir -p results
	$(BIN)/hello > results/hello.txt
	$(BIN)/vector_add_cpu > results/vector_add_cpu.csv
	$(BIN)/vector_add_gpu > results/vector_add_gpu.csv
	$(BIN)/pinned_memory > results/pinned_memory.txt
	$(BIN)/sgemm > results/sgemm_1024.txt
	$(BIN)/conv2d > results/conv2d_cublas.txt

clean:
	rm -rf $(BIN)
.PHONY: all run clean
