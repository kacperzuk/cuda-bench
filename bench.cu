#include <unistd.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 1024*1024*1024/sizeof(unsigned int)  // 1GiB of uints

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void init_states(curandState_t *states, unsigned int seed) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  curand_init(seed,
              index,
              0,
              states+index);
}

__global__ void init_cache(int n, curandState_t *states, unsigned int* cache) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride) 
    cache[i] = curand(states+index);
}

__global__ void bench(int n, unsigned int* cache, unsigned int* r) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  unsigned int s = 0;
  for(int i = index; i < n; i += stride) 
    s += cache[i];
  *r += s;
}

int main (int argc, char *argv[]) {
  if (argc < 2) {
    printf("Provide device number!\n");
    exit(1);
  }

  int gpuNum = atoi(argv[1]);
  gpuErrchk(cudaSetDevice(gpuNum));

  printf("Generating cache...\n");

  int threads = 256;
  int blocks = 1024;

  curandState_t* states;
  unsigned int* gpu_nums;
  cudaMalloc((void**) &states, blocks*threads * sizeof(curandState_t));
  cudaMalloc((void**) &gpu_nums, N * sizeof(unsigned int));
  init_states<<<blocks, threads>>>(states, time(0));
  init_cache<<<blocks, threads>>>(N, states, gpu_nums);

  unsigned int* res;
  unsigned int zero = 0;
  cudaMalloc((void**) &res, sizeof(unsigned int));
  cudaMemcpy(res, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
  unsigned int count = 0;
  unsigned int start = time(0);
  printf("Generated cache.\n");
  while(true) {
    bench<<<blocks, threads>>>(N, gpu_nums, res);
    count++;
    if(time(0) - start > 5) {
      printf(" m - cu%d %d\n", gpuNum, count);
      count = 0;
      start = time(0);
    }
  }
  cudaFree(states);
  cudaFree(gpu_nums);
  cudaFree(res);

  return 0;
}
