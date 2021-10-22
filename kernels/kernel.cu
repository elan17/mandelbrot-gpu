#include <pycuda-complex.hpp>
#include <stdint.h>

typedef pycuda::complex<double> cmplx;

extern "C" __global__ void compute(cmplx *Z, cmplx *C, uint8_t *Iter,
                                   uint16_t height){
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int idx = i*height + j;
    for (int i=0; i < 100; i++){
        Z[idx] = Z[idx] * Z[idx] + C[idx];
        double bounded_d = Z[idx].real() * Z[idx].real() 
                         + Z[idx].imag() * Z[idx].imag();
        bool bounded = bounded_d < 2.0;
        Iter[idx] += bounded;
    }
}