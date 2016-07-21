#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double timeSpecToSeconds(struct timespec* ts)
{
    return ((double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0)*1000;
}

void prefix_suffix_sum(int * q, int * ps, int * ss,int * partial_result1, int * partial_result2, int n, int chunk, int rank, int size) {
		int partial_sum1 = 0, partial_sum2 = 0;
		int j;

		#pragma omp for private(j) schedule(static,chunk)
		for(int i = 0; i < n; i++) {
			j = n - 1 - i;
			//printf("rank=%d i=%d j=%d\n",rank,i,j);
			partial_sum1 += q[i];
			ps[i] = partial_sum1;
			partial_sum2 += q[j];
			//printf("rank=%d i=%d j=%d q[i]=%d, q[j]=%d, ps1=%d ps2=%d\n",rank,i,j,q[i],q[j],partial_sum1,partial_sum2);
			ss[j] = partial_sum2;
		}

		partial_result1[rank] = partial_sum1;
		partial_result2[size - 1 - rank] = partial_sum2;

		//printf("here %d partial result %d\n",rank, partial_sum);
	
		#pragma omp barrier

		#pragma omp single
		{
			for(int i = 1, j = size - 2; i < size; i++, j--) {
				partial_result1[i] += partial_result1[i-1];
				partial_result2[j] += partial_result2[j+1];
				//printf("here %d i=%d partialresult=%d\n",rank, i,partial_result[i]);
			}
		}

		#pragma omp for private(j) schedule(static,chunk)
		for(int i = 0; i < n; i++) {
			j = n - 1 - i;

			if(rank != 0)
				ps[i] += partial_result1[rank-1];
			if(rank != size - 1)
				ss[i] += partial_result2[rank+1];
		}
}

void pmax_smax(int * ps, int * ss, int * pmax, int * smax, int * partial_result1, int * partial_result2, int n, int chunk, int rank, int size) {
		int partial_max1 = 0, partial_max2 = 0;
		int j;

		#pragma omp for private(j) schedule(static,chunk)
		for(int i = 0; i < n; i++) {
			j = n - 1 - i;
			//printf("rank=%d i=%d j=%d\n",rank,i,j);
			partial_max1 = i == 0 ? ss[i] : (ss[i] > partial_max1 ? ss[i] : partial_max1);
			pmax[i] = partial_max1;

			partial_max2 = j == n - 1 ? ps[j] : (ps[j] > partial_max2 ? ps[j] : partial_max2);			
			smax[j] = partial_max2;
			//printf("rank=%d i=%d j=%d q[i]=%d, q[j]=%d, ps1=%d ps2=%d\n",rank,i,j,ss[i],ps[j],partial_max1,partial_max2);
			//printf("rank=%d smax[%d]=%d\n",rank,j,smax[j]);
		}

		partial_result1[rank] = partial_max1;
		partial_result2[size - 1 - rank] = partial_max2;

		//printf("here %d partial result %d\n",rank, partial_sum);

		#pragma omp barrier

		#pragma omp single
		{
			for(int i = 1, j = size - 2; i < size; i++, j--) {
				partial_result1[i] = partial_result1[i] > partial_result1[i-1] ? partial_result1[i] : partial_result1[i-1];
				partial_result2[j] = partial_result2[j] > partial_result2[j+1] ? partial_result2[j] : partial_result2[j+1];
				//printf("pr2[%d]=%d\n",j,partial_result2[j]);
				//printf("here %d i=%d, partialresult=%d\n",rank, i,partial_result[i]);
			}
		}


		#pragma omp for private(j) schedule(static,chunk)
		for(int i = 0; i < n; i++) {
			j = n - 1 - i;

			if(rank != 0 && pmax[i] < partial_result1[rank-1])
				pmax[i] = partial_result1[rank-1];
			if(rank != 0 && smax[j] < partial_result2[size - rank])
				smax[j] = partial_result2[size - rank];	
		}
}

void m_calc(int * ps, int * ss, int * pmax, int * smax, int * q, int * m, int n, int chunk, int rank, int size, int& max_sum) {
	#pragma omp for schedule(static,chunk) reduction(max:max_sum)  
	for(int i = 0; i < n; i++) {
		m[i] = (pmax[i] - ss[i] + q[i]) + (smax[i] - ps[i] + q[i]) - q[i];
		max_sum = i == 0 ? m[i] : (m[i] > max_sum ? m[i] : max_sum);		
	}		
}

void find_values(int * m, int * partial_result, int n, int chunk, int rank, int size, int max_sum, int& maior, int& menor, int& total) {
	int count = 0;
	int control = 0;	

	#pragma omp for schedule(static,chunk)
	for(int i = 0; i < n; i++) {
		control++;
		//printf("%d %d %d\n",rank,m[i],max_sum);
		if(m[i] == max_sum)
			count++;
		else
			count = 0;
		m[i] = count;

		//printf("%d %d %d\n",rank,control,chunk);

		if(control == chunk) {
			partial_result[rank] = m[i];
			//printf("%d %d\n",rank,partial_result[rank]);
		}
	}

	int local_rank = rank - 1;
	int toadd = 0;
	while(local_rank >= 0 && partial_result[local_rank] == chunk) {
		toadd += partial_result[local_rank--];
	}
	if(local_rank >= 0) {
		toadd += partial_result[local_rank];
	}
	//printf("rank=%d toadd=%d\n",rank,toadd);

	bool tostop = false;

	#pragma omp for schedule(static,chunk)
	for(int i = 0; i < n; i++) {
		if(!tostop && m[i] != 0)
			m[i] += toadd;
		else
			tostop = true;
	}

	#pragma omp for schedule(static,chunk) reduction(+:total) reduction(max:maior) reduction(min:menor)
	for(int i = 0; i < n; i++) {
		control++;
		if(m[i] != 0 && i != n - 1 && m[i+1] == 0) {
			total++;
			if(m[i] > maior)
				maior = m[i];
			if(m[i] < menor)
				menor = m[i];
		}

		if(i == n - 1 && m[i] != 0) {
			total++;
			if(m[i] > maior)
				maior = m[i];
			if(m[i] < menor)
				menor = m[i];			
		}
	}	
}

int main() {
	int n;
	scanf("%d",&n);

	int *q, *ps, *ss, *pmax, *smax, *m;
	q = (int*)malloc(n*sizeof(int));
	ps = (int*)malloc(n*sizeof(int));
	ss = (int*)malloc(n*sizeof(int));
	pmax = (int*)malloc(n*sizeof(int));
	smax = (int*)malloc(n*sizeof(int));	
	m = (int*)malloc(n*sizeof(int));

	for(int i = 0; i < n; i++) {
		scanf("%d",&q[i]);
	}

	/*for(int i = 0; i < n; i++) {
		printf("%d ",q[i]);
	}
	printf("\n");*/


	timespec time1, time2;

	clock_gettime(CLOCK_MONOTONIC, &time1);

	/*OpenMP variables*/
	int * partial_result1;
	int * partial_result2;

	int max_sum = 0;

	int maior = INT_MIN;
	int menor = INT_MAX;
	int total = 0;

	#pragma omp parallel default (shared)
	{
		
		int rank = omp_get_thread_num();
		int size = omp_get_num_threads();
		int chunk = (int)(n/size);

		#pragma omp single
		{
			partial_result1 = (int*)malloc(size*sizeof(int));
			partial_result2 = (int*)malloc(size*sizeof(int));
			
		}
		
		#pragma omp barrier


		#pragma omp master
		{
			//Time taking starts
			clock_gettime(CLOCK_MONOTONIC, &time1);
		}

		prefix_suffix_sum(q,ps,ss,partial_result1,partial_result2,n,chunk,rank,size);

		//printf("\n");

		pmax_smax(ps,ss,pmax,smax,partial_result1,partial_result2,n,chunk,rank,size);

		//printf("\n");

		m_calc(ps,ss,pmax,smax,q,m,n,chunk,rank,size,max_sum);

		//printf("\n");

		find_values(m,partial_result1,n,chunk,rank,size,max_sum,maior,menor,total);

		/*#pragma omp single
		{
			for(int i = 0; i < n; i++) {
				printf("%d ",ps[i]);
			}
			printf("\n");

			for(int i = 0; i < n; i++) {
				printf("%d ",ss[i]);
			}
			printf("\n");

			for(int i = 0; i < n; i++) {
				printf("%d ",smax[i]);
			}
			printf("\n");

			for(int i = 0; i < n; i++) {
				printf("%d ",pmax[i]);
			}
			printf("\n");

			for(int i = 0; i < n; i++) {
				printf("%d ",m[i]);
			}
			printf("\n");

			printf("max_sum=%d\n",max_sum);
		}

		printf("rank=%d total=%d maior=%d menor=%d\n",rank,total,maior,menor);*/

		#pragma omp master
		{
			//Time taking ends
			clock_gettime(CLOCK_MONOTONIC, &time2);
		}

		#pragma omp single
		{
			free(partial_result1);
			free(partial_result2);
		}		
	}

	printf("%.9f\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));
	
	free(q);
	free(ps);
	free(ss);
	free(pmax);
	free(smax);
	free(m);

	return 0;
}
