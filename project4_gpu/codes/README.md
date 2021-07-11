### Files

pointer.c, pointer: correct the inaccurate codes in Question 1.

addition_cpu.cu, addition_cpu: implement the vector addition on CPU.

addition_gpu.cu, addition_gpu: implement the vector addition on GPU.

time.txt, plot_time.py, plot_time.png: compare the execute time for the CPU and GPU runs.

Makefile: execute all the C and CUDA codes.

### Run

```bash
make
./pointer
./addition_cpu [vector length]
./addition_gpu [vector length]
```

