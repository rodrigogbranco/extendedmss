#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

double timeSpecToSeconds(struct timespec* ts)
{
    return ((double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0)*1000;
}

int main() {
	int n;
	scanf("%d",&n);

	int *q, *ps, *ss, *pmax, *smax, *m, *binary_transform, *segscan;
	q = (int*)malloc(n*sizeof(int));
	ps = (int*)malloc(n*sizeof(int));
	ss = (int*)malloc(n*sizeof(int));
	pmax = (int*)malloc(n*sizeof(int));
	smax = (int*)malloc(n*sizeof(int));	
	m = (int*)malloc(n*sizeof(int));
	binary_transform = (int*)malloc(n*sizeof(int));
	segscan = (int*)malloc(n*sizeof(int));

	for(int i = 0; i < n; i++) {
		scanf("%d",&q[i]);
	}

	timespec time1, time2, time3;

	clock_gettime(CLOCK_MONOTONIC, &time1);

	for(int i = 0; i < n; i++) {
		ps[i] = i == 0 ? q[i] : q[i] + ps[i-1];
	}

	for(int i = n-1; i >= 0; i--) {
		ss[i] = i == n - 1 ? q[i] : q[i] + ss[i+1];
		smax[i] = i == n - 1 ? ps[i] : (ps[i] > smax[i+1] ? ps[i] : smax[i+1]);
	}

	int max_sum;
	for(int i = 0; i < n; i++) {
		pmax[i] = i == 0 ? ss[i] : (ss[i] > pmax[i-1] ? ss[i] : pmax[i-1]);

		m[i] = (pmax[i] - ss[i] + q[i]) + (smax[i] - ps[i] + q[i]) - q[i];

		max_sum = i == 0 ? m[i] : (m[i] > max_sum ? m[i] : max_sum);
	}

	//Here we have the maximum subarray sum: max_sum
	clock_gettime(CLOCK_MONOTONIC, &time2);

	for(int i = 0; i < n; i++) {
		binary_transform[i] = m[i] == max_sum ? -1 : 0;

		if(i != 0 && binary_transform[i] == binary_transform[i-1])
			segscan[i] = segscan[i-1] + 1;
		else
			segscan[i] = 1;
	}

	int maior = INT_MIN;
	int menor = INT_MAX;
	int total = 0;
	int count = 0;

	int k = -1;
	for(int i = 0; i < n; i++) {
		segscan[i] = segscan[i] & binary_transform[i];

		if(k >= 0 && segscan[k] > 0 && segscan[i] == 0) {
			total++;
			maior = segscan[k] > maior ? segscan[k] : maior;
			menor = segscan[k] < menor ? segscan[k] : menor;			
		}
		k++;
	}

	clock_gettime(CLOCK_MONOTONIC, &time3);

	//printf("%.9f %.9f = mss=%d max=%d min=%d total=%d\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1),
	//	timeSpecToSeconds(&time3) - timeSpecToSeconds(&time2),max_sum,maior,menor,total);
	printf("%.9f\n",timeSpecToSeconds(&time3) - timeSpecToSeconds(&time1));
	//printf("%.9f %.9f\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1),timeSpecToSeconds(&time3) - timeSpecToSeconds(&time1));
	//printf("maxsum=%d maior=%d menor=%d total=%d\n",max_sum,maior,menor,total);
	
	free(q);
	free(ps);
	free(ss);
	free(pmax);
	free(smax);
	free(m);
	free(binary_transform);
	free(segscan);

	return 0;
}
