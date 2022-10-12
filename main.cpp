#include "mpi.h"
#include <chrono>
#include <iostream>
#include <random>
using namespace std;

void Flip(double *&B, int dim){

double temp=0.0; 
  for (int i=0;i<dim;i++){
	for (int j=i+1;j<dim;j++)
	        temp=B[i*dim+j],
	        B[i*dim+j]=B[j*dim+i],
		B[j*dim+i]=temp;
  }
}


double fRand(double fMin, double fMax)
{
     double f = (double)rand() / RAND_MAX;
     return fMin + f * (fMax - fMin);
}

int main() {
	int N = 4;
	double *A;
	double *B;
	double *C;
	long int ns;
	A = new double[N*N];
	B = new double[N*N];
	C = new double[N*N];
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			A[i*N + j] = fRand(0.0, 10.0);
			B[i*N + j] = fRand(0.0, 10.0);
		}
	}
	int ProcRank;
	int ProcNum;

	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	int Size = N;
	int dim = Size;
	int i, j, k, p, ind;
	double temp;
	MPI_Status Status;
	int ProcPartSize = dim/ProcNum;
	int ProcPartElem = ProcPartSize*dim;
	double* bufA = new double[ProcPartElem];
	double* bufB = new double[ProcPartElem];
	double* bufC = new double[ProcPartElem];
	
	int ProcPart = dim/ProcNum;
	int part = ProcPart*dim;
	

	auto start_time = std::chrono::steady_clock::now();
	if (ProcRank == 0) {
		Flip(B, Size);
	}

	
	MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, part, MPI_DOUBLE, bufB, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	temp=0.0;

	for (i=0; i<ProcPartSize; i++) {
		for (j=0; j<ProcPartSize; j++){
			for (k=0; k< dim; k++){
				temp += bufA[i*dim + k]*bufB[j*dim+k];
			}
			bufC[i*dim + j + ProcPartSize*ProcRank] = temp;
			temp = 0.0;
		}
	}

	int NextProc;
	int PrevProc;

	for (p=1; p< ProcNum; p++) {
		NextProc=ProcRank + 1;
		if (NextProc == ProcNum){
			NextProc=0;
		}
		PrevProc=ProcRank - 1;
		if (PrevProc == -1){
			PrevProc = ProcNum - 1;
		}

		MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
		temp=0.0;
		for (i=0; i<ProcPartSize; i++) {
			for (j=0; j<ProcPartSize; j++){
				for (k=0; k< dim; k++){
					temp += bufA[i*dim + k]*bufB[j*dim+k];
				}
				if (ProcRank - p >=0 ){
					ind = ProcRank - p;
				}
				else {
					ind = ProcNum - p + ProcRank;
				}
				bufC[i*dim +j + ind*ProcPartSize] = temp;
				temp=0.0;
			}
		}
	}

	MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	auto end_time = std::chrono::steady_clock::now();
	auto elapsed_ns = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	ns = elapsed_ns.count();
	

	delete []bufA;
	delete []bufB;
	delete []bufC;
	MPI_Finalize();
	if (ProcRank==0){

	/*for (int i =0; i<N; i++){
		for (int j = 0; j< N; j++){
			cout << C[i*N +j] << " ";
		}
		cout << "" << endl;
	}*/
		cout << ns << "ms" << endl;
	}
	
	return 0;

}

