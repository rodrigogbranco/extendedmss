#include "cuda_util.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <thrust/scan.h>
#include <climits>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <omp.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/host_vector.h>
#include <numeric>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define THREAD_NUM 1024
#define BLOCKS_SCHED 2
#define SIZE_WARP 32

/*Definição da operação de máximo, usado nas operações de prefixo e sufixo máximo*/
thrust::maximum<int> max_op;
thrust::minimum<int> min_op;
thrust::plus<int> plus_op;

thrust::bit_and<int> bit_and_op;

/*for use _1 function in thrust::for_each function*/
using namespace thrust::placeholders;

typedef thrust::device_vector<int>::iterator Iterator; 

struct calcM : public thrust::unary_function<thrust::tuple<int,int,int,int,int>,int>
{
    	__host__ __device__
        float operator()(const thrust::tuple<int,int,int,int,int>& a) const
        {
		//0 = d_q, 1 = d_psum, 2 = d_ssum, 3 = d_pmax, 4 = d_smax
		//m[i] = (pmax[i] - ss[i] + q[i]) + (smax[i] - ps[i] + q[i]) - q[i];		
		return (thrust::get<3>(a) - thrust::get<2>(a) + thrust::get<0>(a)) +
		       (thrust::get<4>(a) - thrust::get<1>(a) + thrust::get<0>(a)) -
		       thrust::get<0>(a);
        }
};


/*for transform: m[i] == global_max ? -1 : 0;*/
struct binaryTransform
{
	const int constant;

	binaryTransform(int _constant) : constant(_constant) {}

	__host__ __device__
	void operator()( int& e) const 
	{
		e = e == constant ? -1 : 0;
	}
};

__global__ void findRelatedSolutions(int * v, int partition, int dev_id, int boundary, int * device_max, int * device_min, int * device_total) {
	int tid = threadIdx.x  + blockIdx.x * blockDim.x;
	int total_p = blockDim.x * gridDim.x;

	unsigned short int idWarp = threadIdx.x >> 5;
	unsigned short int idtW = threadIdx.x % SIZE_WARP;
	int valueNeighbor;

	int myvalue = 0;
	int next = 0;

	int mymax = INT_MIN;
	int mymin = INT_MAX;
	int mytotal = 0;

	extern __shared__ int array[];
	int* block_max = (int*)&array[0*blockDim.x];
	int* block_min = (int*)&array[1*blockDim.x];
	int* block_total = (int*)&array[2*blockDim.x];

	block_max[threadIdx.x] = INT_MIN;
	block_min[threadIdx.x] = INT_MAX;
	block_total[threadIdx.x] = 0;

	__syncthreads();

	#pragma unroll 128
	for(int i = tid; i < partition; i += total_p) {
		myvalue = v[i];
		next = i < partition - 1 ? v[i+1] : 0;

		if(myvalue > 0 && next == 0) {
			if((i != partition - 1) || (i == partition - 1 && boundary == 0)) {
				mytotal++;
				mymax = myvalue > mymax ? myvalue : mymax;
				mymin = myvalue < mymin ? myvalue : mymin;
			}
		}
	}

	block_max[threadIdx.x] 	 = mymax;
	block_min[threadIdx.x]	 = mymin;
	block_total[threadIdx.x] = mytotal;

	__syncthreads();

//	if(threadIdx.x == 0) {
//		mymax = INT_MIN;
//		mymin = INT_MAX;
//		mytotal = 0;
//		for(int i = 0; i < blockDim.x; i++) {
//			mytotal += block_total[i];
//			mymax = block_max[i] > mymax ? block_max[i] : mymax;
//			mymin = block_min[i] < mymin ? block_min[i] : mymin;
//		}
//
//		device_max[blockIdx.x] = mymax;
//		device_min[blockIdx.x] = mymin;
//		device_total[blockIdx.x] = mytotal;
//	}

//	mytotal = block_total[threadIdx.x];
//	mymax = block_max[threadIdx.x];
//    mymin = block_min[threadIdx.x];

	int half_block = blockDim.x >> 1;
	for (int half = half_block; half > 0; half>>=1) {//log

		if(threadIdx.x < half){

			mytotal = block_total[threadIdx.x];
			mymax = block_max[threadIdx.x];
			mymin = block_min[threadIdx.x];

			valueNeighbor = block_max[threadIdx.x + half];
			if(mymax < valueNeighbor){
				mymax = valueNeighbor;
				block_max[threadIdx.x] = mymax;
			}

			valueNeighbor = block_min[threadIdx.x + half];
			if(mymin > valueNeighbor){
				mymin = valueNeighbor;
				block_min[threadIdx.x] = mymin;
			}

			valueNeighbor = block_total[threadIdx.x + half];
			mytotal += valueNeighbor;
			block_total[threadIdx.x] = mytotal;
		}

		__syncthreads();
	}

	if(idWarp == 0 && idtW == 0)
		device_max[blockIdx.x] = block_max[0];

	if(idWarp == 1 && idtW == 0)
		device_min[blockIdx.x] = block_min[0];

	if(idWarp == 2 && idtW == 0)
		device_total[blockIdx.x] = block_total[0];

//	#pragma unroll 4
//	for (int i = 16; i > 0; i>>=1) { //log
//		valueNeighbor = __shfl_down(mymax, i);
//		if(mymax < valueNeighbor){
//			mymax = valueNeighbor;
//		}
//
//		valueNeighbor = __shfl_down(mymin, i);
//		if(mymin > valueNeighbor){
//			mymin = valueNeighbor;
//		}
//
//		valueNeighbor = __shfl_down(mytotal, i);
//		mytotal += valueNeighbor;
//	}
//
//	if(idtW == 0){
//		block_max[idWarp] = mymax;
//		block_min[idWarp] = mymin;
//		block_total[idWarp] = mytotal;
//	}
//
//	__syncthreads();

//	if(idWarp == 0){
//		mytotal = block_total[idtW];
//		mymax = block_max[idtW];
//	    mymin = block_min[idtW];
//
//		#pragma unroll 4
//		for (int i = 16; i > 0; i>>=1) { //log
//			valueNeighbor = __shfl_down(mymax, i);
//
//			if(mymax < valueNeighbor){
//				mymax = valueNeighbor;
//			}
//
//			valueNeighbor = __shfl_down(mymin, i);
//			if(mymin > valueNeighbor){
//				mymin = valueNeighbor;
//			}
//
//			valueNeighbor = __shfl_down(mytotal, i);
//			mytotal += valueNeighbor;
//		}
//
//		if(idtW == 0){
//			block_max[idWarp] = mymax;
//			block_min[idWarp] = mymin;
//			block_total[idWarp] = mytotal;
//		}
//	}
//
//	__syncthreads();
//
//	if(idWarp == 0 && idtW == 0)
//		device_max[blockIdx.x] = mymax;
//
//	if(idWarp == 1 && idtW == 0)
//		device_min[blockIdx.x] = block_min[0];
//
//	if(idWarp == 2 && idtW == 0)
//		device_total[blockIdx.x] = block_total[0];
}

int main() {
	
	/*Leitura da entrada*/
	int n;
	scanf("%d",&n);	

	int * input = (int *)malloc(n*sizeof(int));

	//only to check
	int * psum = (int *)malloc(n*sizeof(int));
	int * ssum = (int *)malloc(n*sizeof(int));
	int * pmax = (int *)malloc(n*sizeof(int));
	int * smax = (int *)malloc(n*sizeof(int));
	int * m    = (int *)malloc(n*sizeof(int));
	int * segmentedscan    = (int *)malloc(n*sizeof(int));

	for(int i = 0; i < n; i++) {
		scanf("%d",&input[i]);
	}		

	int devCount;
	HANDLE_ERROR( cudaGetDeviceCount(&devCount));

	thrust::host_vector<int> spawn_right(devCount);
	thrust::host_vector<int> spawn_left(devCount);

	thrust::host_vector<int> local_max(devCount);
	thrust::host_vector<int> local_min(devCount);
	thrust::host_vector<int> local_total(devCount);	

	#pragma omp parallel num_threads(devCount) default(shared)
	{
		cudaEvent_t start,stop,stop2;
		float time1, time2;
		const int dev_id = omp_get_thread_num();

		HANDLE_ERROR( cudaSetDevice(dev_id) );

		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, dev_id);


//		for(int j=0;j<devCount;j++) {
//			if(dev_id != j) {
//				int access;
//				cudaDeviceCanAccessPeer(&access,dev_id,j);
//				if(access) {
//					cudaDeviceEnablePeerAccess(j,0);
//				}
//			}
//		}

		int partition = (int)(n/devCount);
		if(dev_id == devCount -1) {
			if(n % devCount != 0)
				partition += n % devCount;
		}

		thrust::device_vector<int> d_q(partition);
		thrust::device_vector<int> d_psum(partition);
		thrust::device_vector<int> d_ssum(partition);
		thrust::device_vector<int> d_smax(partition);
		thrust::device_vector<int> d_pmax(partition);
		thrust::device_vector<int> d_m(partition);

		thrust::device_vector<int> d_segmentedscan(partition);

		int num_threads = THREAD_NUM;
		int num_blocks = devProp.multiProcessorCount*BLOCKS_SCHED;

		dim3 threadsPorBloco(num_threads);
		dim3 blocosPorGrid(num_blocks);

		thrust::device_vector<int> d_result_max(num_blocks);
		thrust::device_vector<int> d_result_min(num_blocks);
		thrust::device_vector<int> d_result_total(num_blocks);

		thrust::device_vector<int>::iterator iter;

		thrust::fill(thrust::device, d_segmentedscan.begin(), d_segmentedscan.end(), 1);
		thrust::fill(thrust::device, d_result_max.begin(), d_result_max.end(), INT_MIN);
		thrust::fill(thrust::device, d_result_min.begin(), d_result_min.end(), INT_MAX);
		thrust::fill(thrust::device, d_result_total.begin(), d_result_total.end(), 0);

		thrust::copy(input + partition*dev_id, input + partition*dev_id + partition, d_q.begin());

		int correct1, correct2 = 0;
		int k, l;
		int global_max;		

		/*Tomada de tempo na GPU-0*/
		if(dev_id == 0) {
			HANDLE_ERROR( cudaEventCreate(&start) );
			HANDLE_ERROR( cudaEventCreate(&stop) );
			HANDLE_ERROR( cudaEventCreate(&stop2) );
			HANDLE_ERROR( cudaEventRecord(start, 0) );
		}

		thrust::inclusive_scan(d_q.begin(), d_q.end(), d_psum.begin());
		thrust::inclusive_scan(d_q.rbegin(), d_q.rend(), d_ssum.rbegin());

		/*IMPROVE AllToAll - Now it's using host memory*/
		spawn_right[dev_id] = d_psum[partition - 1];
		spawn_left[dev_id] = d_ssum[0];

		HANDLE_ERROR( cudaDeviceSynchronize() );
		#pragma omp barrier

		correct1 = 0; correct2 = 0;
		k = 0, l = devCount - 1;
		while(k < dev_id || l > dev_id) {
			if(k < dev_id)
				correct1 += spawn_right[k];

			if(l > dev_id)
				correct2 += spawn_left[l];

			k++; l--;
		}

		if(dev_id != 0)
			thrust::for_each(d_psum.begin(), d_psum.end(), _1 += correct1);

		if(dev_id != devCount - 1)
			thrust::for_each(d_ssum.rbegin(), d_ssum.rend(), _1 += correct2);
			

		thrust::inclusive_scan(d_psum.rbegin(), d_psum.rend(), d_smax.rbegin(), max_op);
		thrust::inclusive_scan(d_ssum.begin(), d_ssum.end(), d_pmax.begin(), max_op);

		/*IMPROVE AllToAll -Now it's using host memory*/
		spawn_right[dev_id] = d_pmax[partition - 1];
		spawn_left[dev_id] = d_smax[0];

		HANDLE_ERROR( cudaDeviceSynchronize() );
		#pragma omp barrier

		correct1 = INT_MIN; correct2 = INT_MIN;
		k = 0, l = devCount - 1;
		while(k < dev_id || l > dev_id) {
			if(k < dev_id)
				correct1 = spawn_right[k] >  correct1 ? spawn_right[k] : correct1;

			if(l > dev_id)
				correct2 = spawn_left[l] > correct2 ? spawn_left[l] : correct2;

			k++; l--;
		}

		if(dev_id != 0) {
			//Optimize it!!
			if(correct1 > d_pmax[0]) {
				d_pmax[0] = correct1;
				thrust::inclusive_scan(d_pmax.begin(), d_pmax.end(), d_pmax.begin(), max_op);
			}
		}

		if(dev_id != devCount - 1) {
			//Optimize it!!
			if(correct2 > d_smax[partition - 1]) {
				d_smax[partition - 1] = correct2;
				thrust::inclusive_scan(d_smax.rbegin(), d_smax.rend(), d_smax.rbegin(), max_op);
			}			
		}

		/*Cálculo do vetor M*/
	        thrust::transform(thrust::device, 
			thrust::make_zip_iterator(make_tuple(d_q.begin(),d_psum.begin(),d_ssum.begin(),d_pmax.begin(),d_smax.begin())),
			thrust::make_zip_iterator(make_tuple(d_q.end(),d_psum.end(),d_ssum.end(),d_pmax.end(),d_smax.end())),		
			       d_m.begin(),
			       calcM() );

		local_max[dev_id] = thrust::reduce(thrust::device, d_m.begin(), d_m.end(), INT_MIN, max_op);

		#pragma omp barrier

		global_max = thrust::reduce(thrust::host, local_max.begin(), local_max.end(), INT_MIN, max_op);

		/*Finalizando a tomada de tempo - Primeira parte - Encontrar o máximo*/
		if(dev_id == 0) {
			HANDLE_ERROR( cudaEventRecord(stop, 0) );
			HANDLE_ERROR( cudaEventSynchronize(stop) );
			HANDLE_ERROR( cudaEventElapsedTime(&time1, start, stop) );
		}

		thrust::for_each(d_m.begin(), d_m.end(), binaryTransform(global_max));

		thrust::inclusive_scan_by_key(thrust::device, d_m.begin(), d_m.end(), d_segmentedscan.begin(), d_segmentedscan.begin());

		thrust::transform(thrust::device, d_m.begin(), d_m.end(), d_segmentedscan.begin(), d_segmentedscan.begin(), bit_and_op);

		/*IMPROVE AllToAll -Now it's using host memory*/
		spawn_right[dev_id] = d_segmentedscan[partition - 1];

		HANDLE_ERROR( cudaDeviceSynchronize() );
		#pragma omp barrier

		correct1 = 0;
		l = dev_id - 1;
		while(l >= 0) {
			if(spawn_right[l] > 0) {
				correct1 += spawn_right[l];
			}
			if(spawn_right[l] == partition)
				l--;
			else
				break;
		}

		iter = thrust::find(thrust::device, d_segmentedscan.begin(), d_segmentedscan.end(), 0);
		thrust::for_each(d_segmentedscan.begin(), iter, _1 += correct1);

		int * segscan = thrust::raw_pointer_cast(d_segmentedscan.data());
		int * dmax = thrust::raw_pointer_cast(d_result_max.data());
		int * dmin = thrust::raw_pointer_cast(d_result_min.data());
		int * dtotal = thrust::raw_pointer_cast(d_result_total.data());

		/*IMPROVE AllToAll -Now it's using host memory*/
		spawn_left[dev_id] = d_segmentedscan[0];

		HANDLE_ERROR( cudaDeviceSynchronize() );
		#pragma omp barrier

		int boundary = dev_id < devCount - 1 ? spawn_left[dev_id+1] : 0;
		findRelatedSolutions<<<blocosPorGrid,threadsPorBloco, 3*num_threads*sizeof(int)>>>(segscan,partition, dev_id, boundary, dmax, dmin, dtotal);
		CudaCheckError();

		local_max[dev_id] = thrust::reduce(thrust::device, d_result_max.begin(), d_result_max.end(), INT_MIN, max_op);
		local_min[dev_id] = thrust::reduce(thrust::device, d_result_min.begin(), d_result_min.end(), INT_MAX, min_op);
		local_total[dev_id] = thrust::reduce(thrust::device, d_result_total.begin(), d_result_total.end(), 0, plus_op);

		HANDLE_ERROR( cudaDeviceSynchronize() );
		#pragma omp barrier

		int maximum = INT_MIN;
		int minimum = INT_MAX;
		int total = 0;
		for(int i = 0; i < devCount; i++) {
			maximum = local_max[i] > maximum ? local_max[i] : maximum;
			minimum = local_min[i] < minimum ? local_min[i] : minimum;
			total += local_total[i];
		}
		

		/*Finalizando a tomada de tempo - Segunda parte - Problemas Relacionados*/
		if(dev_id == 0) {
			HANDLE_ERROR( cudaEventRecord(stop2, 0) );
			HANDLE_ERROR( cudaEventSynchronize(stop2) );
			HANDLE_ERROR( cudaEventElapsedTime(&time2, stop, stop2) );
			printf("%.9f %.9f = mss=%d max=%d min=%d total=%d\n",time1,time2,global_max,maximum,minimum,total);
		}

		thrust::copy(d_psum.begin(), d_psum.end(), psum + partition*dev_id);
		thrust::copy(d_ssum.begin(), d_ssum.end(), ssum + partition*dev_id);
		thrust::copy(d_pmax.begin(), d_pmax.end(), pmax + partition*dev_id);
		thrust::copy(d_smax.begin(), d_smax.end(), smax + partition*dev_id);
		thrust::copy(d_m.begin(), d_m.end(), m + partition*dev_id);
		thrust::copy(d_segmentedscan.begin(), d_segmentedscan.end(), segmentedscan + partition*dev_id);

		#pragma omp barrier	

	}

	/*printf("input: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",input[i]);
	}
	printf("\n");

	printf("psum: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",psum[i]);
	}
	printf("\n");

	printf("ssum: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",ssum[i]);
	}
	printf("\n");

	printf("pmax: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",pmax[i]);
	}
	printf("\n");

	printf("smax: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",smax[i]);
	}
	printf("\n");

	printf("m: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",m[i]);
	}
	printf("\n");

	printf("local_max: \n");
	for(int i = 0; i < devCount; i++) {
		printf("%d ",local_max[i]);
	}
	printf("\n");

	printf("local_min: \n");
	for(int i = 0; i < devCount; i++) {
		printf("%d ",local_min[i]);
	}
	printf("\n");

	printf("local_total: \n");
	for(int i = 0; i < devCount; i++) {
		printf("%d ",local_total[i]);
	}
	printf("\n");

	printf("seg-scan: \n");
	for(int i = 0; i < n; i++) {
		printf("%d ",segmentedscan[i]);
	}
	printf("\n");

	printf("boundary: \n");
	for(int i = 0; i < devCount; i++) {
		printf("%d ",spawn_left[i]);
	}
	printf("\n");*/



	free(input);

	free(psum);
	free(ssum);
	free(pmax);
	free(smax);
	free(m);
	free(segmentedscan);

	return 0;
}
