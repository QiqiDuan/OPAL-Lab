#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>



/**
 * HOST: handle the CUDA Errors.
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
 * DEVICE: add two vectors.
 */
__global__ void devAddVect( const double *dev_vect_a, const double *dev_vect_b,
		double *dev_vect_c, const unsigned vect_length ) {
	unsigned tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( tidx >= vect_length ) {
		return ;
	}
	dev_vect_c[ tidx ] = dev_vect_a[ tidx ] + dev_vect_b[ tidx ];
	__syncthreads();
}



/**
 * DEVICE: add two vectors.
 */
__global__ void devAddVectArbitraryLength( const double *dev_vect_a, const double *dev_vect_b,
		double *dev_vect_c, const unsigned vect_length ) {
	unsigned tidx = threadIdx.x + blockIdx.x * blockDim.x;
	while ( tidx < vect_length ) {
		dev_vect_c[ tidx ] = dev_vect_a[ tidx ] + dev_vect_b[ tidx ];
		tidx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}



/**
 * HOST: whether two double-precision floating-point values are approximately equal or not.
 */
int is_equal( double x, double y ) {
	return fabs( x - y ) < 1e-6 ? 1 : 0;
}



void call_devAddVect( void ) {
	/* add two vectors */
	printf( "\n*********************\n* add two vectors *\n*********************\n" );
	int ind_dev = 0;
	HANDLE_CUDA_ERROR( cudaSetDevice( ind_dev ) );

	unsigned vect_length = 10000000;
	size_t vect_length_bytes = vect_length * sizeof( double );
	double *vect_a = NULL, *vect_b = NULL, *vect_c = NULL;
	vect_a = ( double * ) malloc( vect_length_bytes );
	vect_b = ( double * ) malloc( vect_length_bytes );
	vect_c = ( double * ) malloc( vect_length_bytes );
	if ( vect_a == NULL || vect_b == NULL || vect_c == NULL ) {
		fprintf( stderr, "\nHOST ERROR :: cannot allocate enough memory.\n" );
		exit( EXIT_FAILURE );
	}
	for ( unsigned i = 0; i < vect_length; i++ ) {
		vect_a[ i ] = i;
		vect_b[ i ] = 2 * i;
	}

	double *dev_vect_a, *dev_vect_b, *dev_vect_c;
	HANDLE_CUDA_ERROR( cudaMalloc( ( double ** ) &dev_vect_a, vect_length_bytes ) );
	HANDLE_CUDA_ERROR( cudaMalloc( ( double ** ) &dev_vect_b, vect_length_bytes ) );
	HANDLE_CUDA_ERROR( cudaMalloc( ( double ** ) &dev_vect_c, vect_length_bytes ) );

	HANDLE_CUDA_ERROR( cudaMemcpy( dev_vect_a, vect_a, vect_length_bytes, cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( dev_vect_b, vect_b, vect_length_bytes, cudaMemcpyHostToDevice ) );

	unsigned num_thread_per_block = 512;
	unsigned num_block_per_grid = (vect_length + num_thread_per_block - 1 ) / num_thread_per_block;
	time_t gpu_run_time_start, gpu_run_time_end;
	time( &gpu_run_time_start );
	devAddVect <<< num_block_per_grid, num_thread_per_block >>> (
			dev_vect_a, dev_vect_b, dev_vect_c, vect_length );
	time( &gpu_run_time_end );
	double gpu_run_time = difftime( gpu_run_time_end, gpu_run_time_start );
	HANDLE_CUDA_ERROR( cudaPeekAtLastError() );
	HANDLE_CUDA_ERROR( cudaMemcpy( vect_c, dev_vect_c, vect_length_bytes, cudaMemcpyDeviceToHost ) );

	printf( "Totoal GPU run time: %lf.\n", gpu_run_time );
	for ( unsigned i = 0; i < vect_length; i++ ) {
		if ( is_equal( vect_c[ i ], vect_a[ i ] + vect_b[ i ] ) == 0 ) {
			printf( "\nERROR :: final results are not right.\n" );
			printf( ">> i = %u :: vect_c[ i ] = %lf vs. vect_a[ i ] + vect_b[ i ] = %lf.\n",
					i, vect_c[ i ], vect_a[ i ] + vect_b[ i ] );
			exit( EXIT_FAILURE );
		} else {
			if ( i == 0 || i == ( vect_length / 2) || i == ( vect_length - 1 ) ) {
				printf( ">> i = %u :: vect_c (%.2lf) == vect_a (%.2lf) + vect_b (%.2lf).\n",
						i, vect_c[ i ], vect_a[ i ], vect_b[ i ] );
			}
		}
	}

	HANDLE_CUDA_ERROR( cudaFree( dev_vect_a ) );
	HANDLE_CUDA_ERROR( cudaFree( dev_vect_b ) );
	HANDLE_CUDA_ERROR( cudaFree( dev_vect_c ) );
	free( vect_a );
	free( vect_b );
	free( vect_c );
	cudaDeviceReset();
}



void call_devAddVectArbitraryLength( void ) {
	/* add two vectors with arbitrary length */
		printf( "\n*********************\n* add two vectors with arbitrary length *\n*********************\n" );
		int ind_dev = 1;
		HANDLE_CUDA_ERROR( cudaSetDevice( ind_dev) );

		unsigned vect_length = 100000000;
		size_t vect_length_bytes = vect_length * sizeof( double );
		double *vect_a = NULL, *vect_b = NULL, *vect_c = NULL;
		vect_a = ( double * ) malloc( vect_length_bytes );
		vect_b = ( double * ) malloc( vect_length_bytes );
		vect_c = ( double * ) malloc( vect_length_bytes );
		if ( vect_a == NULL || vect_b == NULL || vect_c == NULL ) {
			fprintf( stderr, "\nHOST ERROR :: cannot allocate enough memory.\n" );
		}
		for ( unsigned i = 0; i < vect_length; i++ ) {
			vect_a[ i ] = i;
			vect_b[ i ] = 2 * i;
		}

		double *dev_vect_a, *dev_vect_b, *dev_vect_c;
		HANDLE_CUDA_ERROR( cudaMalloc( ( double ** ) &dev_vect_a, vect_length_bytes ) );
		HANDLE_CUDA_ERROR( cudaMalloc( ( double ** ) &dev_vect_b, vect_length_bytes ) );
		HANDLE_CUDA_ERROR( cudaMalloc( ( double ** ) &dev_vect_c, vect_length_bytes ) );

		HANDLE_CUDA_ERROR( cudaMemcpy( dev_vect_a, vect_a, vect_length_bytes, cudaMemcpyHostToDevice ) );
		HANDLE_CUDA_ERROR( cudaMemcpy( dev_vect_b, vect_b, vect_length_bytes, cudaMemcpyHostToDevice ) );

		unsigned num_thread_per_block = 1024;
		unsigned num_block_per_grid = 1024;
		time_t gpu_run_time_start, gpu_run_time_end;
		time( &gpu_run_time_start );
		devAddVectArbitraryLength <<< num_block_per_grid, num_thread_per_block >>> (
				dev_vect_a, dev_vect_b, dev_vect_c, vect_length );
		time( &gpu_run_time_end );
		double gpu_run_time = difftime( gpu_run_time_end, gpu_run_time_start );
		HANDLE_CUDA_ERROR( cudaPeekAtLastError() );
		HANDLE_CUDA_ERROR( cudaMemcpy( vect_c, dev_vect_c, vect_length_bytes, cudaMemcpyDeviceToHost ) );

		printf( "Totoal GPU run time: %lf.\n", gpu_run_time );
		for ( unsigned i = 0; i < vect_length; i++ ) {
			if ( is_equal( vect_c[ i ], vect_a[ i ] + vect_b[ i ] ) == 0 ) {
				printf( "\nError :: final results are not right.\n" );
				printf( ">> i = %u :: vect_c[ i ] = %lf vs. vect_a[ i ] + vect_b[ i ] = %lf.\n",
						i, vect_c[ i ], vect_a[ i ] + vect_b[ i ] );
				exit( EXIT_FAILURE );
			} else {
				if ( i == 0 || i == ( vect_length / 2 ) || i == ( vect_length - 1 ) ) {
					printf( ">> i = %u :: vect_c (%.2lf) == vect_a (%.2lf) + vect_b (%.2lf).\n",
							i, vect_c[ i ], vect_a[ i ], vect_b[ i ] );
				}
			}
		}

		HANDLE_CUDA_ERROR( cudaFree( dev_vect_a ) );
		HANDLE_CUDA_ERROR( cudaFree( dev_vect_b ) );
		HANDLE_CUDA_ERROR( cudaFree( dev_vect_c ) );
		free( vect_a );
		free( vect_b );
		free( vect_c );
		cudaDeviceReset();
}



int main( void ) {
	call_devAddVect();
	call_devAddVectArbitraryLength();
}
