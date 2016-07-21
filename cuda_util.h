#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdio.h>

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }  
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

inline void __cudaSafeCall( cudaError err, const char *file, const int line ){
	#ifdef CUDA_ERROR_CHECK
		if ( cudaSuccess != err )
		{
			fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
					 file, line, cudaGetErrorString( err ) );
			exit( -1 );
		}
	#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line ){
	#ifdef CUDA_ERROR_CHECK
		cudaError err = cudaGetLastError();
		if ( cudaSuccess != err )
		{
			fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
					 file, line, cudaGetErrorString( err ) );
			exit( -1 );
		}

	#endif

    return;
}

#endif
