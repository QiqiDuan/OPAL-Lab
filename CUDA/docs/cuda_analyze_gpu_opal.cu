#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * HOST: Handle the CUDA Errors.
 */
#define HANDLE_CUDA_ERROR( cuda_expression ) { assertGpuError( ( cuda_expression ), __FILE__, __LINE__ ); }
inline void assertGpuError( cudaError_t error_index, const char *error_file, const unsigned int error_line ) {
	if ( error_index != cudaSuccess ) {
		fprintf( stderr, "\n\n\n***\nCUDA ERROR :: %s [LINE %d] ---> %s.\n***\n\n\n",
				error_file, error_line, cudaGetErrorString( error_index ) );
		cudaDeviceReset();
		exit( -1 );
	}
}



int main ( void ) {
	int num_gpu;

	HANDLE_CUDA_ERROR( cudaGetDeviceCount( &num_gpu ) );
	printf( "\n***\n* Total number of GPU devices currently available: %d.\n***\n", num_gpu );

	cudaDeviceProp gpu_info;
	for ( int ind_gpu = 0; ind_gpu < num_gpu; ind_gpu++ ) {
		HANDLE_CUDA_ERROR( cudaGetDeviceProperties( &gpu_info, ind_gpu ) );
		printf( "\n*************************************************\n" );
		printf( "*** General Info. for GPU Device index: %d.\n", ind_gpu );
		printf( "* Device name: %s.\n", gpu_info.name );
		printf( "* Compute mode: %d.\n", gpu_info.computeMode );
		printf( "* Compute capability: %d.%d.\n", gpu_info.major, gpu_info.minor );
		printf( "* Kernel execution timeout: %s.\n", 
			gpu_info.kernelExecTimeoutEnabled ? "enabled" : "disabled" );

		printf( "*** Memory Info.:\n" );
		printf( "* Total global memory: %zu (~= %5.2lf GB).\n",
			gpu_info.totalGlobalMem,
			( double ) gpu_info.totalGlobalMem / pow( 2.0, 30 ) );
		printf( "* Shared memory per block: %zu (~= %5.2lf KB).\n", 
			gpu_info.sharedMemPerBlock,
			( double ) gpu_info.sharedMemPerBlock / pow( 2.0, 10 ) );

		printf( "*** MP Info.:\n" );
		printf( "* Multi-processor num: %d.\n", gpu_info.multiProcessorCount );
		printf( "* Max threads per blocks: %d.\n", gpu_info.maxThreadsPerBlock );
		printf( "* Max thread dim: (%d, %d, %d).\n",
			gpu_info.maxThreadsDim[ 0 ],
			gpu_info.maxThreadsDim[ 1 ],
			gpu_info.maxThreadsDim[ 2 ] );
		printf( "* Max grid dim: (%d, %d, %d).\n",
			gpu_info.maxGridSize[ 0 ], 
			gpu_info.maxGridSize[ 1 ], 
			gpu_info.maxGridSize[ 2 ] );
	}
}
