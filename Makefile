all: build

build: CUFFT_nvidia_3d

 

CUFFT_nvidia_3d.o:CUFFT_nvidia_3d.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

CUFFT_nvidia_3d: CUFFT_nvidia_3d.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	 

run: build
	$(EXEC) ./CUFFT_nvidia_3d

clean:
	rm -f CUFFT_nvidia_3d CUFFT_nvidia_3d.o
	 

clobber: clean