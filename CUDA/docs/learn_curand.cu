/**
 * Generate Uniformly Distributed Random Numbers via the CUDA cuRAND Library on the NVIDIA GPU.
 */

#include <stdio.h>
#include <math.h>
#include <time.h>

/**
 * On Ubuntu, The below header files are generally located in:
 *      /usr/local/cuda/include
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>



/**
 * HOST: Handle the CUDA Errors.
 */
#define HCE( cuda_expression ) { assertGpuError( ( cuda_expression ), __FILE__, __LINE__ ); }
inline void assertGpuError( cudaError_t error_index, 
    const char *error_file, const unsigned error_line ) {
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
__global__ void devSetRngState( unsigned total_num_threads, 
        unsigned rng_seed, curandState *dev_rgn_state ) {
    unsigned tidx = threadIdx.x + blockIdx.x * blockDim.x;
    while ( tidx < total_num_threads ) {
        curand_init( rng_seed, tidx, 0, dev_rgn_state + tidx );
        tidx += blockDim.x * gridDim.x;
    }
}



/**
 * DEVICE: Generate Uniformly Distributed Random Numbers for GPU.
 */
__global__ void devGenUniformRand( unsigned total_num_threads, 
        curandState *dev_rgn_state, float *dev_rand_samples ) {
    unsigned tidx = threadIdx.x + blockIdx.x * blockDim.x;
    while ( tidx < total_num_threads ) {
        *( dev_rand_samples + tidx ) = curand_uniform( dev_rgn_state + tidx );
        tidx += blockDim.x * gridDim.x;
    }
}



/**
 * HOST: calculate the average value for a numeric array.
 */
float avg( const float *array, const unsigned array_length ) {
    float asum = 0.0;
    for ( unsigned ind_array = 0; ind_array < array_length; ind_array++ ) {
        asum += *( array + ind_array );
    }
    return asum / array_length;
}



int main( void ) {
    printf("\n*******\n* Generate Uniformly Distributed Random Numbers "
        "via the CUDA cuRAND Library on the NVIDIA GPU\n*******\n");
    srand( ( unsigned ) time( NULL ) );

    const unsigned NUM_BLOCKS_PER_GRID = 64;
    const unsigned NUM_THREADS_PER_BLOCK = 64;

    printf("\n*** check random numbers on parallel threads ***\n");
    curandState *dev_rgn_state;
    float *dev_rand_samples;
    float *rand_samples;
    for ( unsigned ind_num_threads = 1; ind_num_threads <= 10000000; ind_num_threads *= 10 ) {
        printf( "ind_num_threads = %u --->\n", ind_num_threads );
        HCE( cudaMalloc( ( curandState ** ) &dev_rgn_state, 
            ind_num_threads * sizeof( curandState ) ) );
        HCE( cudaMalloc( ( float ** ) &dev_rand_samples, 
            ind_num_threads * sizeof( float ) ) );
        devSetRngState <<< NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK >>> ( 
            ind_num_threads, ( unsigned ) rand(), dev_rgn_state );
        rand_samples = ( float * ) malloc( ind_num_threads * sizeof( float ) );
        devGenUniformRand <<< NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK >>> ( 
            ind_num_threads, dev_rgn_state, dev_rand_samples );
        HCE( cudaMemcpy( rand_samples, dev_rand_samples, 
            ind_num_threads * sizeof( float ), cudaMemcpyDeviceToHost ) );
        for ( unsigned ind_elem = 0; ind_elem < ind_num_threads; ind_elem++ ) {
            if ( rand_samples[ ind_elem ] > 1.0 || rand_samples[ ind_elem ] <= 0.0 ) {
                fprintf( stderr, "\n\n\nERROR >> cannot generate correct random numbers.\n\n\n" );
            }
        }
        printf( "rand_samples[%i] = %5.3lf, [%u] = %5.3lf, [%u] = %5.3lf && Avg = %7.5lf\n",
            0, rand_samples[ 0 ],
            ind_num_threads / 2, rand_samples[ ind_num_threads / 2 ], 
            ind_num_threads - 1, rand_samples[ ind_num_threads - 1 ],
            avg( rand_samples, ind_num_threads ) );
        HCE( cudaFree( dev_rgn_state ) );
        HCE( cudaFree( dev_rand_samples ) );
        free( rand_samples );
    }

    printf("\n*** check random numbers on different iterations ***\n");
    unsigned num_threads = 1000;
    curandState *dev_rgn_state2;
    float *dev_rand_samples2;
    float *rand_samples2;
    HCE( cudaMalloc( ( curandState ** ) &dev_rgn_state2, num_threads * sizeof( curandState ) ) );
    HCE( cudaMalloc( ( float ** ) &dev_rand_samples2, num_threads * sizeof( float ) ));
    rand_samples2 = ( float * ) malloc( num_threads * sizeof( float ) );
    devSetRngState <<< NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK >>> ( 
            num_threads, ( unsigned ) rand(), dev_rgn_state2 );
    float avg_fst = 0.0, avg_half = 0.0, avg_end = 0.0;
    unsigned num_iter = 5000;
    for ( unsigned ind_iter = 1; ind_iter <= num_iter; ind_iter++ ) {
        devGenUniformRand <<< NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK >>> ( 
            num_threads, dev_rgn_state2, dev_rand_samples2 );
        HCE( cudaMemcpy( rand_samples2, dev_rand_samples2, 
            num_threads * sizeof( float ), cudaMemcpyDeviceToHost ) );
        avg_fst += rand_samples2[ 0 ];
        avg_half += rand_samples2[ num_threads / 2 ];
        avg_end += rand_samples2[ num_threads - 1 ];
        for ( unsigned ind_elem = 0; ind_elem < num_threads; ind_elem++ ) {
            if ( rand_samples2[ ind_elem ] > 1.0 || rand_samples2[ ind_elem ] <= 0.0 ) {
                fprintf( stderr, "\n\n\nERROR >> cannot generate correct random numbers.\n\n\n" );
            }
        }
        if ( ind_iter == 1 || ind_iter % 500 == 0 || ind_iter == num_iter ) {
            printf( "ind_iter = %u ---> rand_samples[%i] = %5.3lf, [%u] = %5.3lf, [%u] = %5.3lf\n",
            ind_iter,
            0, rand_samples2[ 0 ],
            num_threads / 2, rand_samples2[ num_threads / 2 ], 
            num_threads - 1, rand_samples2[ num_threads - 1 ]);
        }
    }
    printf("avg_fst = %5.3lf && avg_half = %5.3lf && avg_end = %5.3lf\n", 
        avg_fst / num_iter, avg_half / num_iter, avg_end / num_iter );
    HCE( cudaFree( dev_rgn_state2 ) );
    HCE( cudaFree( dev_rand_samples2 ) );
    free( rand_samples2 );
}
