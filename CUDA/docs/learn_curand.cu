#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>



/**
 * HOST: Handle the CUDA Errors.
 */
#define HANDLE_CUDA_ERROR( cuda_expression ) { assertGpuError( ( cuda_expression ), __FILE__, __LINE__ ); }

inline void assertGpuError( cudaError_t error_index, const char *error_file, const unsigned error_line ) {
	if ( error_index != cudaSuccess ) {
		fprintf( stderr, "\n\n\n***\nCUDA ERROR :: %s [LINE %u] ---> %s.\n***\n\n\n",
				error_file, error_line, cudaGetErrorString( error_index ) );
		cudaDeviceReset();
		exit( EXIT_FAILURE );
	}
}



/**
 * DEVICE: Set the RNG State for GPU.
 */
__global__ void devSetRngState( unsigned ini_rng_seed, curandState *dev_rgn_state ) {
    unsigned tidx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init( threadIdx.x + ini_rng_seed, threadIdx.x, 0, dev_rgn_state + tidx );
}



/**
 * DEVICE: Generate the Uniformly Distributed Number for GPU.
 */
__global__ void devGenUniformRand( curandState *dev_rgn_state, float *dev_rand_list ) {
    unsigned tidx = threadIdx.x + blockIdx.x * blockDim.x;
    *( dev_rand_list + tidx ) = curand_uniform( dev_rgn_state + tidx );
}



int main( void ) {
    printf("*** generate the uniformly distributed random number.\n");

    const unsigned NUM_ITER = 10;
    const unsigned NUM_ELEMENTS = 5000000;
    const unsigned NUM_BLOCKS_PER_GRID = 128;
    const unsigned NUM_THREADS_PER_BLOCK = ( NUM_ELEMENTS + NUM_BLOCKS_PER_GRID - 1 ) / NUM_BLOCKS_PER_GRID;
    
    curandState *dev_rgn_state;
    HANDLE_CUDA_ERROR( cudaMalloc( ( curandState ** ) &dev_rgn_state, NUM_ELEMENTS * sizeof( curandState ) ) );
    devSetRngState <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_PER_GRID >>> ( (size_t) time( NULL ), dev_rgn_state );

    float rand_list[ NUM_ELEMENTS ];
    float *dev_rand_list;
    HANDLE_CUDA_ERROR( cudaMalloc( ( float ** ) &dev_rand_list, NUM_ELEMENTS * sizeof( float ) ) );

    float asum = 0.0;
    float asum_list[ NUM_ELEMENTS ];
    for ( unsigned ind_elem = 0; ind_elem < NUM_ELEMENTS; ind_elem++ ) {
        asum_list[ ind_elem ] = 0.0;
    }

    for ( unsigned ind_iter = 0; ind_iter < NUM_ITER; ind_iter++ ) {
	devGenUniformRand <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_PER_GRID >>> ( dev_rgn_state, dev_rand_list );
        HANDLE_CUDA_ERROR( cudaMemcpy( rand_list, dev_rand_list, NUM_ELEMENTS * sizeof( float ), cudaMemcpyDeviceToHost ) );
	
	printf( "Iter Sample :: ind_iter = %u ---> uniform_rand = %lf.\n", ind_iter, rand_list[ NUM_ELEMENTS / 2 ] );	
	
        asum = 0.0;
        for ( unsigned ind_elem = 0; ind_elem < NUM_ELEMENTS; ind_elem++ ) {
            if (rand_list[ ind_elem ] > 1.0 || rand_list[ ind_elem ] <= 0.0 ) {
                fprintf( stderr, "rand_list[ %u ] = %lf.\n", ind_elem, rand_list[ ind_elem ] );
                fprintf( stderr, "ERROR :: cannot correctly generate the uniformly distributed random number.\n" );
                exit( EXIT_FAILURE );
            }
            asum += rand_list[ ind_elem ];
            asum_list[ ind_elem ] += rand_list[ ind_elem ];
        }
        if ( fabs( asum / NUM_ELEMENTS - 0.5 ) > 1e-3 ) {
            printf( "Check :: Avg = %lf.\n", asum / NUM_ELEMENTS );
        }
    }

    for ( unsigned ind_elem = 0; ind_elem < NUM_ELEMENTS; ind_elem++ ) {
	if ( ind_elem == 0 || ind_elem == NUM_ELEMENTS / 2 || ind_elem == NUM_ELEMENTS - 1 ) {
		printf( "Elem Sample :: ind_elem = %u ---> uniform_rand = %lf.\n", ind_elem, rand_list[ ind_elem ] );
	}
    }

    for( unsigned ind_iter = 0; ind_iter < NUM_ITER; ind_iter++ ) {
        if ( fabs( asum_list[ ind_iter ] / NUM_ITER - 0.5 ) > 1e-3 ) {
            printf( "Check :: ind_iter = %u ---> Iter Avg = %lf.\n", ind_iter, asum_list[ ind_iter ] / NUM_ITER );
        }
    }

    HANDLE_CUDA_ERROR( cudaFree( dev_rgn_state ) );
    HANDLE_CUDA_ERROR( cudaFree( dev_rand_list ) );
}
