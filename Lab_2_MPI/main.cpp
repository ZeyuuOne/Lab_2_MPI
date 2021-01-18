#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

int m, n, groupSize, myId, numProcs;
float* A, * L, * U, * localA, * mainRow;
int i, j, k, l, lower, upper;
double start, end;
FILE* inFile, * outFile;
MPI_Status status;

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);

	if (myId == 0) {
		inFile = fopen("LU.in", "r");
		fscanf(inFile, "%d %d", &m, &n);
		A = (float*)malloc(sizeof(float) * m * m);
		for (i = 0; i < m; i++) for (j = 0; j < m; j++) fscanf(inFile, "%f", A + i * m + j);
		fclose(inFile);
		L = (float*)malloc(sizeof(float) * m * m);
		U = (float*)malloc(sizeof(float) * m * m);
		start = MPI_Wtime();
	}

	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	groupSize = m / numProcs;
	if (m % numProcs != 0) groupSize++;
	
	localA = (float*)malloc(sizeof(float) * groupSize * m);
	mainRow = (float*)malloc(sizeof(float) * m);

	if (myId == 0){
		for (i = 0; i < groupSize; i++)for (j = 0; j < m; j++) localA[i * m + j] = A[i * m + j];
		for (i = 1; i < numProcs - 1; i++) MPI_Send(&A[i * groupSize * m], m * groupSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		if (numProcs > 1) MPI_Send(&A[(numProcs - 1) * groupSize * m], m * (m - (numProcs - 1) * groupSize), MPI_FLOAT, numProcs - 1, 0, MPI_COMM_WORLD);
	}
	else if (myId != numProcs - 1){
		MPI_Recv(localA, m * groupSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	}
	else {
		MPI_Recv(localA, m * (m - (numProcs - 1) * groupSize), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	}

	for (j = 0; j < numProcs; j++){
		upper = (j == numProcs - 1) ? (m - (numProcs - 1) * groupSize) : groupSize;

		for (i = 0; i < upper; i++) {
			if (myId == j) {
				lower = j * groupSize + i;
				for (k = lower; k < m; k++) mainRow[k] = localA[i * m + k];
				MPI_Bcast(mainRow, m, MPI_FLOAT, j, MPI_COMM_WORLD);
			}
			else {
				lower = j * groupSize + i;
				MPI_Bcast(mainRow, m, MPI_FLOAT, j, MPI_COMM_WORLD);
			}

			if (myId <= j) {
				for (k = i + 1; k < upper; k++) {
					localA[k * m + lower] = localA[k * m + lower] / mainRow[lower];
					for (l = lower + 1; l < m; l++) localA[k * m + l] = localA[k * m + l] - mainRow[l] * localA[k * m + lower];
				}
			}
			else {
				for (k = i; k < upper; k++) {
					localA[k * m + lower] = localA[k * m + lower] / mainRow[lower];
					for (l = lower + 1; l < m; l++) localA[k * m + l] = localA[k * m + l] - mainRow[l] * localA[k * m + lower];
				}
			}
		}
	}

	if (myId == 0){
		for (i = 0; i < groupSize; i++) for (j = 0; j < m; j++) A[i * m + j] = localA[i * m + j];
	}
	else if (myId != numProcs - 1) {
		MPI_Send(localA, m * groupSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Send(localA, m * (m - (numProcs - 1) * groupSize), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}

	if (myId == 0) {
		for (i = 1; i < numProcs - 1; i++){
			MPI_Recv(localA, m * groupSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			for (j = 0; j < groupSize; j++) for (k = 0; k < m; k++) A[(i * groupSize + j) * m + k] = localA[j * m + k];
		}
		if (numProcs > 1) {
			MPI_Recv(localA, m * (m - (numProcs - 1) * groupSize), MPI_FLOAT, numProcs - 1, 0, MPI_COMM_WORLD, &status);
			for (j = 0; j < m - (numProcs - 1) * groupSize; j++) for (k = 0; k < m; k++) A[(i * groupSize + j) * m + k] = localA[j * m + k];
		}
	}

	if (myId == 0)
	{
		end = MPI_Wtime();

		for (i = 0; i < m; i++) {
			for (j = 0; j < m; j++) {
				L[i * m + j] = j == i ? 1 : 0;
				U[i * m + j] = 0;
				if (i > j) L[i * m + j] = A[i * m + j];
				else U[i * m + j] = A[i * m + j];
			}
		}

		outFile = fopen("LU.out", "w");
		for (i = 0; i < m; i++)
		{
			for (j = 0; j < m; j++) fprintf(outFile, "%f\t", L[i * m + j]);
			fprintf(outFile, "\n");
		}
		fprintf(outFile, "\n");
		for (i = 0; i < m; i++)
		{
			for (j = 0; j < m; j++) fprintf(outFile, "%f\t", U[i * m + j]);
			fprintf(outFile, "\n");
		}
		fclose(outFile);

		printf("M: %d, Thread Num: %d, Time: %fs\n", m, numProcs, end - start);
	}

	MPI_Finalize();
	free(localA);
	free(mainRow);
	return(0);
}