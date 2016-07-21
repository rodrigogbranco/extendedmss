#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define ROOT 0
#define FALSE 0
#define TRUE 1

int main(int argc, char** argv) {
	
	/*some usual variables*/
	int rank, size, n;
	int i, j, *input;

	/*MPI Initialization*/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*Generating a MPI Communication with reverse rank order*/
	int * process_ranks = (int*) malloc(size*sizeof(int));

	for(i = 0, j = size - 1; i < size; i++, j--) {
		process_ranks[i] = j;
	}

	MPI_Group group_world;
	MPI_Group reverse_group;
	MPI_Comm reverse_comm;

	MPI_Comm_group(MPI_COMM_WORLD, &group_world);

	MPI_Group_incl(group_world, size, process_ranks, &reverse_group);
	MPI_Comm_create(MPI_COMM_WORLD, reverse_group, &reverse_comm);

	free(process_ranks);

	/*Reading the input...*/
	if(rank == ROOT) {
		scanf("%d",&n);
	}
	
	MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	if(rank == ROOT) {
		input = (int *)malloc(n*sizeof(int));
		for(i = 0; i < n; i++) {
			scanf("%d",&input[i]);
		}
	}

	/*creating auxiliary vectors*/
	int partition = (int)(n/size);

	int * q = (int *)malloc(partition*sizeof(int));
	int * ps = (int*)malloc(partition*sizeof(int));
	int * ss = (int*)malloc(partition*sizeof(int));
	int * pmax = (int*)malloc(partition*sizeof(int));
	int * smax = (int*)malloc(partition*sizeof(int));	
	int * m = (int*)malloc(partition*sizeof(int));

	int * frontier = malloc(size*sizeof(int));
	int * frontier_start = malloc(size*sizeof(int));
	int * partial_result = (int*)malloc(partition*sizeof(int));
	partial_result[0] = 0;

	/*variables to auxiliary reductions*/
	int reduce1, reduce2;

	/*variables to computate extended problems*/
	int maior = INT_MIN;
	int menor = INT_MAX;
	int total = 0;

	int global_maior;
	int global_menor;
	int global_total;

	/*stores maxSum*/
	int max_sum = INT_MIN;
	int global_max;

	/*splitting the input*/
	MPI_Scatter(input, partition, MPI_INT, q, partition, MPI_INT, ROOT, MPI_COMM_WORLD);

	if(rank == ROOT) {
		free(input);
	}

	/*Here the algorithm starts*/
	MPI_Barrier(MPI_COMM_WORLD);
	double start_p1 = MPI_Wtime();

	/*calculation of prefix and suffix sum*/
	for(i = 0, j = partition - 1; i < partition; i++, j--) {
		ps[i] = i == 0 ? q[i] : q[i] + ps[i-1];
		ss[j] = j == partition - 1 ? q[j] : q[j] + ss[j+1];
	}

	reduce1 = ps[partition-1];
	reduce2 = ss[0];

	MPI_Exscan(&reduce1,&reduce1,1, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
	MPI_Exscan(&reduce2,&reduce2,1, MPI_INT, MPI_SUM, reverse_comm);

	for(i = 0, j = partition - 1; i < partition; i++, j--) {
		if(rank > ROOT)
			ps[i] += reduce1;

		if(rank < size - 1)
			ss[j] += reduce2;
	}

	/*calculation of prefix and suffix propagation*/
	for(i = 0, j = partition - 1; i < partition; i++, j--) {
		pmax[i] = i == 0 ? ss[i] : (ss[i] > pmax[i-1] ? ss[i] : pmax[i-1]);		
		smax[j] = j == partition - 1 ? ps[j] : (ps[j] > smax[j+1] ? ps[j] : smax[j+1]);
	}

	reduce1 = pmax[partition-1];
	reduce2 = smax[0];

	MPI_Exscan(&reduce1,&reduce1,1, MPI_INT, MPI_MAX,MPI_COMM_WORLD);
	MPI_Exscan(&reduce2,&reduce2,1, MPI_INT, MPI_MAX, reverse_comm);

	for(i = 0, j = partition - 1; i < partition; i++, j--) {
		if(rank > ROOT)
			pmax[i] = reduce1 > pmax[i] ? reduce1 : pmax[i];

		if(rank < size - 1)
			smax[j] = reduce2 > smax[j] ? reduce2 : smax[j];
	}

	for(i = 0; i < partition; i++) {	
		m[i] = (pmax[i] - ss[i] + q[i]) + (smax[i] - ps[i] + q[i]) - q[i];
		max_sum = m[i] > max_sum ? m[i] : max_sum;
	}

	/*Here we've got the solution for basic problem*/
	MPI_Allreduce(&max_sum,&global_max,1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	/*Here we start the algorithm for extended problems*/
	//MPI_Barrier(MPI_COMM_WORLD);
	//double start_p2 = MPI_Wtime();

	for(i = 0; i < partition; i++) {
		if(m[i] == global_max) {
			partial_result[i] = i == 0 ? 1 : partial_result[i-1] + 1;
		}
		else {
			partial_result[i] = 0;
		}
	}

	MPI_Allgather(&partial_result[partition-1], 1, MPI_INT, frontier, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(&partial_result[0], 1, MPI_INT, frontier_start, 1, MPI_INT, MPI_COMM_WORLD);

	int toadd = 0;
	for(i = rank - 1; i >= 0; i--) {
		if(frontier[i] > 0)
			toadd += frontier[i];

		if(frontier[i] != partition)
			break;
	}

	int mustadd = FALSE;
	if(toadd > 0 && partial_result[0] > 0) {
		mustadd = TRUE;
	}
	
	for(i = 0; i < partition; i++) {
		if(mustadd && partial_result[i] > 0) {
			partial_result[i] += toadd;
		}
		else {
			mustadd = FALSE;
		}

		if(partial_result[i] == 1) {
			total++;
		}
		
		maior = partial_result[i] > maior ? partial_result[i] : maior;

		if((i != partition -1 && partial_result[i] > 0 && partial_result[i+1] == 0) ||
		   (i == partition -1 && partial_result[i] > 0 && rank != size - 1 && frontier_start[rank+1] == 0) ||
		   (i == partition -1 && partial_result[i] > 0 && rank == size - 1)) {
			menor = (partial_result[i] < menor && partial_result[i] > 0) ? partial_result[i] : menor;			
		}
	}

	MPI_Reduce(&maior,&global_maior,1, MPI_INT, MPI_MAX, ROOT, MPI_COMM_WORLD);
	MPI_Reduce(&menor,&global_menor,1, MPI_INT, MPI_MIN, ROOT, MPI_COMM_WORLD);
	MPI_Reduce(&total,&global_total,1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

	/*Here we finish our algorithm*/
	MPI_Barrier(MPI_COMM_WORLD);
	double start_p3 = MPI_Wtime();


	if(rank == ROOT) {
		printf("%.9f\n",(start_p3-start_p1)*1000);
		//printf("%.9f %.9f\n",(start_p2-start_p1)*1000,(start_p3-start_p1)*1000);
		//printf("maxsum=%d maior=%d menor=%d total=%d\n",global_max,global_maior,global_menor,global_total);
	}

	/*for(j = 0; j < size; j++) {
		if(rank == j) {
			for(i = 0; i < partition; i++) {
				printf("rank=%d i=%d ps[i]=%d\n",rank,i,ps[i]);
			}

			for(i = 0; i < partition; i++) {
				printf("rank=%d i=%d ss[i]=%d\n",rank,i,ss[i]);
			}

	
			for(i = 0; i < partition; i++) {
				printf("rank=%d i=%d pmax[i]=%d\n",rank,i,pmax[i]);
			}

			for(i = 0; i < partition; i++) {
				printf("rank=%d i=%d smax[i]=%d\n",rank,i,smax[i]);
			}

			for(i = 0; i < partition; i++) {
				printf("rank=%d i=%d m[i]=%d\n",rank,i,m[i]);
			}

			for(i = 0; i < partition; i++) {
				printf("rank=%d i=%d pr[i]=%d\n",rank,i,partial_result[i]);
			}

			for(i = 0; i < size; i++) {
				printf("rank=%d i=%d frontier[i]=%d\n",rank,i,frontier[i]);
			}			
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}*/

	free(q);
	free(ps);
	free(ss);
	free(pmax);
	free(smax);
	free(m);
	free(frontier);
	free(frontier_start);
	free(partial_result);

	MPI_Finalize();
	return 0;
}
