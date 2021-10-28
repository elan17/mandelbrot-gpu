#include <pycuda-complex.hpp>
#include <stdint.h>

typedef pycuda::complex<double> cmplx;

extern "C" __global__ void compute(uint8_t *Iter, cmplx pos,
                                   double zoom, uint8_t iterations, 
                                   uint8_t length){
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint idx = i + j*length;
    Iter[idx] = 0;
    cmplx Z = cmplx(0.0, 0.0);
    cmplx C = pos + cmplx(i, j) * zoom;
    for (int i=0; i < iterations; i++){
        Z = Z * Z + C;
        double bounded_d = Z.real() * Z.real() 
                         + Z.imag() * Z.imag();
        bool bounded = bounded_d < 2.0;
        Iter[idx] += bounded;
    }
}