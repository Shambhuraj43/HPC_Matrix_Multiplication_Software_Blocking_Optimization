#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "cblas.h"
#include "papi.h"
#include <x86intrin.h>


#define NUM_EVENTS 4
#define UNROLL (2)
#define BLOCKSIZE (32)

//Function Declarations
void fillArray(double*);
void printMatrix(double* matrix);
void resetMatrix(double*);
void compareMatrices(double *matrix1, double *matrix2, int size);
double calculateGFLOPS(double, int);


//matrix multiplication functions
void dgemmIJK();

void avx_dgemmIJK(int size);
void avx_dgemmIJK_with_unrolling(int size);
void software_blocking_dgemm(int size);
void do_block(int size, int si, int sj, int sk, double* A, double*B, double *C);


//Global Variables

int N;

int sizeArray[] = {1,2,3,4,5,6,7,8,9,10,11,12}; //{64,128,256,512,1028,2048 }

double * matrixA;
double * matrixB;
double * matrixC;
double * matrixC1;


// int main() {
//
// 	//sradnd initialization
// 	srand(time(NULL));
//
// 	//please comment out all other functions and run only one at a time
//
//
//
//
//
//
// 			for(int i=0;i<6;i++){
//
//
// 				//Do not comment this out////
// 							N = sizeArray[i];
// 				//***********************////
//
//
//
//
// 				//avx_dgemmIJK(sizeArray[i]);
//
// 				//avx_dgemmIJK_with_unrolling(sizeArray[i]);
// 				software_blocking_dgemm(sizeArray[i]);
//
// 			}
//
//
//
// 			//call directly just the function NOT IN FOR LOOP
// 			//dgemmIJK();
//
//
//
// 	//system("pause");
// 	return 0;
// }
//
//
// void printMatrix(double * matrix) {
//
//
// 	printf("**************************************************************************************************************\n\n");
// 	for (int i = 0; i < N; i++) {
//
// 		for (int j = 0; j < N; j++) {
//
// 			printf("%lf  |", matrix[i*N + j]);
// 		}
//
// 		printf("\n");
// 	}
// 	printf("**************************************************************************************************************\n\n");
// }
//
//
// //Function fill array
//
// void  fillArray(double * matrix) {
//
// 	for (int i = 0; i < N; i++) {
//
// 		for (int j = 0; j < N; j++) {
//
// 			double r = (double)rand() / RAND_MAX * 2.0;      //float in range -1 to 1
// 			matrix[i*N + j] = r;
// 		}
//
// 	}
//
// }
//
//
//
// //Function to initialize the matrix to 0
// void resetMatrix(double* matrix) {
//
// 	for (int i = 0; i < N; i++) {
//
// 		for (int j = 0; j < N; j++) {
//
// 			matrix[i*N + j] = 0;
// 		}
//
// 	}
//
//
// }
//
// //function to calculate GFLOPS
//
// double	calculateGFLOPS(double time, int n) {
//
// 	double gflop = (2.0 * n*n*n) / (time * 1e+6);
//
// 	return gflop;
// }
//
// //Function to compare Matrices
//
// void compareMatrices(double *matrix1, double *matrix2,int size) {
//
// 	for (int i = 0; i < size;i ++) {
//
// 		for (int j = 0; j < size; j++) {
//
// 			if (matrix1[i*size + j] == matrix2[i*size + j]) {
// 				//do nothing
//
// 			}
// 			else {
// 				return;
// 			}
// 		}
// 	}
//
// 	printf("\nMatrices are equal!\n");
// }
//
//
// void do_block(int size, int si, int sj, int sk, double* A, double* B, double *C){
// 	for ( int i = si; i < si+BLOCKSIZE; i+=UNROLL*4 ){
//
// 	   for ( int j = sj; j < sj+BLOCKSIZE; j++ ) {
// 	        __m256d c[UNROLL]; //instead of C[1].
//
// 	         for ( int x = 0; x < UNROLL; x++ ){ //compiler flag “–O3” will help you duplicate the loop body.
// 	            c[x] = _mm256_load_pd(C+i*size+x*4+j);
// 	          } //x goes from 0 to UNROLL – 1.
//
// 	         for( int k = sk; k < sk+BLOCKSIZE; k++ )
// 	         {
// 	              __m256d b = _mm256_broadcast_sd(B+k*size+j);
//
// 	               for (int x = 0; x < UNROLL; x++) {//compiler flag “-O3” will help you duplicate the loop body.
// 	                   c[x] = _mm256_add_pd(c[x], //x goes from 0 to UNROLL – 1.
// 	                   _mm256_mul_pd(_mm256_load_pd(A+k+x*4+i*size), b));
// 	               } //end of dot products, whose results are stored in c[0:UNROLL-1].
// 	         }
//
// 	         for ( int x = 0; x < UNROLL; x++ ){ //compiler flag ”-O3” will help you duplicate the loop body.
// 	              _mm256_store_pd(C+i*size+x*4+j, c[x]); //x goes from 0 to UNROLL – 1.
// 	         }
// 	     }
// 	}



	// //matrix multiplication
	//
	// 		for (int i = si; i < si+BLOCKSIZE; ++i)
	// 		{
	//
	// 			 for (int j = sj; j < sj+BLOCKSIZE; ++j)
	// 			 {
	// 					 double cij = C[i*size+j];/* cij = C[i][j] */
	//
	// 					 for( int k = sk; k < sk+BLOCKSIZE; k++ )
	// 					 {
	//
	// 								cij += A[i*size+k] * B[k*size+j];/* cij+=A[i][k]*B[k][j] */
	//
	// 								C[i*size+j] = cij;/* C[i][j] = cij */
	// 					 }
 	// 		   }
	// 		 }
}

// void software_blocking_dgemm(int size){
//
//
// 	//////////////************   PAPI CODE   ************//////////////////////////////
//
// 		int Events[NUM_EVENTS] = { PAPI_FP_OPS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_LST_INS }; //level 1 cache misses
//
// 		int EventSet = PAPI_NULL;
//
// 		long long values[NUM_EVENTS];
//
// 		int retval;
//
//
// 		/* Initialize the Library*/
//
// 		retval = PAPI_library_init(PAPI_VER_CURRENT);
//
//
// 		/* Allocate space for the new eventset and do setup */
// 		retval = PAPI_create_eventset(&EventSet);
//
//
// 		/* Add Flops and total cycles to the eventset*/
// 		retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);
//
// 		//////////////************   PAPI CODE   ************////////////////////////////
//
//
// 		double * A;
// 		double * B;
// 		double * C;
// 		double * C1;
//
// 		int alpha, beta;
// 		alpha = beta = 1.0;
//
//
// 		clock_t  start;
// 		clock_t end;
//
// 		double cpu_time_used;
//
// 		double sum = 0;
//
// 		// //memory allocation for array pointers
// 		// 		posix_memalign((void**)&A, 32, size*size*sizeof(double));
// 		// 		posix_memalign((void**)&B, 32, size*size*sizeof(double));
// 		//  	 	posix_memalign((void**)&C, 32, size*size*sizeof(double));
// 		//
// 		// 		posix_memalign((void**)&C1, 32, size*size*sizeof(double));
//
//
// 				A = (double *)(malloc(size*size * sizeof(double)));
// 				B = (double *)(malloc(size*size * sizeof(double)));
// 				C = (double *)(malloc(size*size * sizeof(double)));
// 				C1 = (double *)(malloc(size*size * sizeof(double)));
//
//
// 					//filling the matrix
// 					fillArray(A);
// 					fillArray(B);
//
//
//
// 					//reset matrixC1
// 					resetMatrix(C);
// 					resetMatrix(C1);
//
// 					///////////////////////////////////////////////////////
//
// 							/* Start the counters */
// 							retval = PAPI_start(EventSet);
//
// 					// 	///////////////////////////////////////////////////////
//
//
// 					//reset matrixC
// 								resetMatrix(C);
//
//
//
//
// 			start = clock();
// 		// matrix multiplication
// 		 for (int si = 0; si < size; si += BLOCKSIZE )
// 		 {
//
// 			 for ( int sj = 0; sj < size; sj += BLOCKSIZE )
// 			 {
//
// 				 for ( int sk = 0; sk < size; sk += BLOCKSIZE )
// 				 {
//
// 							do_block(size, si, sj, sk, A, B, C);
// 					 }
//
// 				}
//
// 			}
//
// 			 end = clock();
//
// 	 					cpu_time_used = ((double)(end - start));
//
// 	 					//sum += cpu_time_used;
//
// 	 			/////////////////////////////////////////////////////////////////
//
// 	 				/*Stop counters and store results in values */
//
// 	 				retval = PAPI_stop(EventSet, values);
//
// 	 			////////////////////////////////////////////////////////////////
//
//
//
// 	 				//Matrix Verification Using CBLAS
//
// 	 			//	Computing Matrix Multiplication using CBLAS
// 	 				cblas_dgemm
// 										(	CblasRowMajor,
// 											CblasNoTrans,
// 											CblasNoTrans,
// 											size,
// 											size,
// 											size,
// 											alpha,
// 											A,
// 											size,
// 											B,
// 											size,
// 											beta,
// 											C1,
// 											size);
//
// 	 				// //Verifying the result
// 	 				compareMatrices(C, C1 ,size);
//
//
//
// 	 				printf("**************************************************************************************************************\n");
//
// 	 				printf("Avg execution time with Ctimer\n Size (NxN) %d \t\t %lf\n", size, (cpu_time_used));
// 	 				printf("GFLOPS:\t\t %lf\n",calculateGFLOPS(cpu_time_used,size) );
// 	 				sum = 0;
//
// 	 				printf("______________________________________________________________________________________________________________\n");
//
// 	 				printf("PAPI Data\n");
//
// 	 				for (int ctr = 0; ctr < NUM_EVENTS; ctr++) {
//
// 	 					printf("%lld\n", values[ctr]);
// 	 				}
//
//
//
//
// 	 				printf("**************************************************************************************************************\n");
//
// 	 				//freeing the dynamic memory
// 	 				free(A);
// 	 				free(B);
// 	 				free(C);
// 	 				free(C1);
//
// }



// void avx_dgemmIJK_with_unrolling(int size) {
//
//
// 	//////////////************   PAPI CODE   ************//////////////////////////////
//
// 	int Events[NUM_EVENTS] = { PAPI_FP_OPS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_LST_INS }; //level 1 cache misses
//
// 	int EventSet = PAPI_NULL;
//
// 	long long values[NUM_EVENTS];
//
// 	int retval;
//
//
// 	/* Initialize the Library*/
//
// 	retval = PAPI_library_init(PAPI_VER_CURRENT);
//
//
// 	/* Allocate space for the new eventset and do setup */
// 	retval = PAPI_create_eventset(&EventSet);
//
//
// 	/* Add Flops and total cycles to the eventset*/
// 	retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);
//
// 	//////////////************   PAPI CODE   ************////////////////////////////
//
//
//
// 	int alpha, beta;
// 	alpha = beta = 1.0;
//
//
// 	clock_t  start;
// 	clock_t end;
//
// 	double cpu_time_used;
//
// 	double sum = 0;
//
//
//
//
//
//
// 		//memory allocation for matrices
//
//
// 		posix_memalign((void**)&matrixA, 32, size*size*sizeof(double));
// 		posix_memalign((void**)&matrixB, 32, size*size*sizeof(double));
//  	 	posix_memalign((void**)&matrixC, 32, size*size*sizeof(double));
//
// 		posix_memalign((void**)&matrixC1, 32, size*size*sizeof(double));
//
// 			//filling the matrix
// 			fillArray(matrixA);
// 			fillArray(matrixB);
//
// 			//reset matrixC1
// 			resetMatrix(matrixC1);
//
//
// 	///////////////////////////////////////////////////////
//
// 		/* Start the counters */
// 		retval = PAPI_start(EventSet);
//
// 	///////////////////////////////////////////////////////
//
//
//
// 			//reset matrixC
// 			resetMatrix(matrixC);
//
// 			start = clock();
//
// 			//matrix multiplication
//
// 			for ( int i = 0; i < size; i+=UNROLL*4 ){
//
// 				 for ( int j = 0; j < size; j++ ) {
// 				 			__m256d c[UNROLL]; //instead of C[1].
//
// 							 for ( int x = 0; x < UNROLL; x++ ){ //compiler flag “–O3” will help you duplicate the loop body.
// 							 		c[x] = _mm256_load_pd(matrixC+i*size+x*4+j);
// 						 		} //x goes from 0 to UNROLL – 1.
//
// 							 for( int k = 0; k < size; k++ )
// 							 {
// 							 			__m256d b = _mm256_broadcast_sd(matrixB+k*size+j);
//
// 										 for (int x = 0; x < UNROLL; x++) {//compiler flag “-O3” will help you duplicate the loop body.
// 												 c[x] = _mm256_add_pd(c[x], //x goes from 0 to UNROLL – 1.
// 												 _mm256_mul_pd(_mm256_load_pd(matrixA+k+x*4+i*size), b));
// 										 } //end of dot products, whose results are stored in c[0:UNROLL-1].
// 						 	 }
//
// 							 for ( int x = 0; x < UNROLL; x++ ){ //compiler flag ”-O3” will help you duplicate the loop body.
// 							 			_mm256_store_pd(matrixC+i*size+x*4+j, c[x]); //x goes from 0 to UNROLL – 1.
// 							 }
//
// 			 		 }
//
// 		 }
//
//
//
// 			end = clock();
//
// 			cpu_time_used = ((double)(end - start));
//
// 			//sum += cpu_time_used;
//
//
//
//
// 	/////////////////////////////////////////////////////////////////
//
// 		/*Stop counters and store results in values */
//
// 		retval = PAPI_stop(EventSet, values);
//
// 	////////////////////////////////////////////////////////////////
//
//
//
// 		//Matrix Verification Using CBLAS
//
// 	//	Computing Matrix Multiplication using CBLAS
// 		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, alpha, matrixA, size, matrixB, size, beta, matrixC1, size);
//
// 		// //Verifying the result
// 		compareMatrices(matrixC,matrixC1,size);
//
//
//
// 		printf("**************************************************************************************************************\n");
//
// 		printf("Avg execution time with Ctimer\n Size (NxN) %d \t\t %lf\n", size, (cpu_time_used));
// 		printf("GFLOPS:\t\t %lf\n",calculateGFLOPS(cpu_time_used,size) );
// 		sum = 0;
//
// 		printf("______________________________________________________________________________________________________________\n");
//
// 		printf("PAPI Data\n");
//
// 		for (int ctr = 0; ctr < NUM_EVENTS; ctr++) {
//
// 			printf("%lld\n", values[ctr]);
// 		}
//
//
// 		// printMatrix(matrixA);
// 		// printMatrix(matrixB);
// 		// printMatrix(matrixC);
// 		// printMatrix(matrixC1);
//
// 		printf("**************************************************************************************************************\n");
//
// 		//freeing the dynamic memory
// 		free(matrixA);
// 		free(matrixB);
// 		free(matrixC);
// 		free(matrixC1);
//
//
// }





/*******************************************************************************************************
							Matrix Multiplication using IJK algorithm using AVX to improve performance

********************************************************************************************************/
// void avx_dgemmIJK(int size) {
//
//
// 	//////////////************   PAPI CODE   ************//////////////////////////////
//
// 	int Events[NUM_EVENTS] = { PAPI_FP_OPS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_LST_INS }; //level 1 cache misses
//
// 	int EventSet = PAPI_NULL;
//
// 	long long values[NUM_EVENTS];
//
// 	int retval;
//
//
// 	/* Initialize the Library*/
//
// 	retval = PAPI_library_init(PAPI_VER_CURRENT);
//
//
// 	/* Allocate space for the new eventset and do setup */
// 	retval = PAPI_create_eventset(&EventSet);
//
//
// 	/* Add Flops and total cycles to the eventset*/
// 	retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);
//
// 	//////////////************   PAPI CODE   ************////////////////////////////
//
//
// 	double alpha, beta;
// 	alpha = beta = 1.0;
//
//
// 	clock_t  start;
// 	clock_t end;
//
// 	double cpu_time_used;
//
// 	double sum = 0;
//
//
// 		//memory allocation for matrices
//
//
// 		posix_memalign((void**)&matrixA, 32, size*size*sizeof(double));
// 		posix_memalign((void**)&matrixB, 32, size*size*sizeof(double));
//  	posix_memalign((void**)&matrixC, 32, size*size*sizeof(double));
// 		posix_memalign((void**)&matrixC1, 32, size*size*sizeof(double));
//
// 			//filling the matrix
// 			fillArray(matrixA);
// 			fillArray(matrixB);
//
// 			//reset matrixC1
// 			resetMatrix(matrixC1);
//
//
// 	///////////////////////////////////////////////////////
//
// 		/* Start the counters */
// 		retval = PAPI_start(EventSet);
//
// 	///////////////////////////////////////////////////////
//
// 		for (int ctr2 = 0; ctr2 < 2; ctr2++) {
//
//
//
// 			resetMatrix(matrixA);
// 			resetMatrix(matrixB);
// 			//filling the matrix
// 			fillArray(matrixA);
// 			fillArray(matrixB);
//
// 			//reset matrixC
// 			resetMatrix(matrixC);
//
// 			start = clock();
// 			//matrix multiplication
// 			for (int i = 0; i < size; i+=4) {
// 				for (int j = 0; j < size; j++) {
// 					//double cij = matrixC[i*N + j];
// 						__m256d c = _mm256_load_pd(matrixC+i*size+j);
// 						for (int k = 0; k < size; k++) {
// 							//cij = cij + matrixA[i*N + k] * matrixB[k*N + j];
// 						//matrixC[i*N + j] = cij;
// 							c = _mm256_add_pd(c, _mm256_mul_pd(_mm256_load_pd(matrixA+i*size+k),
// 																					_mm256_broadcast_sd(matrixB+k*size+j)));
// 																					_mm256_store_pd(matrixC+i*size+j, c);
// 					}
// 				}
// 			}
//
// 			end = clock();
//
// 			cpu_time_used = ((double)(end - start));
//
// 			sum += cpu_time_used;
// 		}
//
// 	/////////////////////////////////////////////////////////////////
//
// 		/*Stop counters and store results in values */
//
// 		 retval = PAPI_stop(EventSet, values);
//
// 	////////////////////////////////////////////////////////////////
//
//
//
// 		//Matrix Verification Using CBLAS
//
// 		//Computing Matrix Multiplication using CBLAS
//
// 		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, alpha, matrixA, size, matrixB, size, beta, matrixC1, size);
//
// 		//Verifying the result
// 		compareMatrices(matrixC,matrixC1);
//
//
//
// 		printf("\n**************************************************************************************************************\n\n");
//
// 		printf("Avg execution time with Ctimer\n Size (NxN) %d \t\t %lf\n\n", size, (sum / 3.0));
// 		printf("GFLOPS:\t\t %lf\n\n",calculateGFLOPS(sum,size) );
// 		sum = 0;
//
// 		printf("\n\n______________________________________________________________________________________________________________\n\n");
//
// 		printf("PAPI Data/n");
//
// 		for (int ctr = 0; ctr < NUM_EVENTS; ctr++) {
//
// 			printf("\n/////////////////////////////////////\n");
// 			printf("%lld\n\n", values[ctr]);
// 		}
//
// 		printf("\n**************************************************************************************************************\n\n");
//
// 		//freeing the dynamic memory
// 		free(matrixA);
// 		free(matrixB);
// 		free(matrixC);
// 		free(matrixC1);
//
//
// }







/*******************************************************************************************************
							Matrix Multiplication using IJK algorithm

********************************************************************************************************/
// void dgemmIJK() {
//
//
// 	//////////////************   PAPI CODE   ************//////////////////////////////
//
// 	int Events[NUM_EVENTS] = { PAPI_FP_OPS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_LST_INS }; //level 1 cache misses
//
// 	int EventSet = PAPI_NULL;
//
// 	long long values[NUM_EVENTS];
//
// 	int retval;
//
//
// 	/* Initialize the Library*/
//
// 	retval = PAPI_library_init(PAPI_VER_CURRENT);
//
//
// 	/* Allocate space for the new eventset and do setup */
// 	retval = PAPI_create_eventset(&EventSet);
//
//
// 	/* Add Flops and total cycles to the eventset*/
// 	retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);
//
// 	//////////////************   PAPI CODE   ************////////////////////////////
//
//
//
// 	int alpha, beta;
// 	alpha = beta = 1.0;
//
//
// 	clock_t  start;
// 	clock_t end;
//
// 	double cpu_time_used;
//
// 	double sum = 0;
//
//
// 	for (int ctr1 = 0; ctr1 < 6; ctr1++) {
//
// 		//memory allocation for matrices
//
// 		N = sizeArray[ctr1];
//
// 		//memory allocation for array pointers
// 		matrixA = (double *)(malloc(N*N * sizeof(double)));
// 		matrixB = (double *)(malloc(N*N * sizeof(double)));
// 		matrixC = (double *)(malloc(N*N * sizeof(double)));
//
// 		matrixC1 = (double *)(malloc(N*N * sizeof(double)));
//
// 			//filling the matrix
// 			fillArray(matrixA);
// 			fillArray(matrixB);
//
// 			//reset matrixC1
// 			resetMatrix(matrixC1);
//
// 	///////////////////////////////////////////////////////
//
// 		/* Start the counters */
// 		retval = PAPI_start(EventSet);
//
// 	///////////////////////////////////////////////////////
//
// 		for (int ctr2 = 0; ctr2 < 3; ctr2++) {
//
// 			//reset matrixC
// 			resetMatrix(matrixC);
//
// 			start = clock();
//
// 			//matrix multiplication
//
// 			for (int i = 0; i < N; i++) {
//
// 				for (int j = 0; j < N; j++) {
//
// 					double cij = matrixC[i*N + j];
//
// 					for (int k = 0; k < N; k++) {
//
// 						cij = cij + matrixA[i*N + k] * matrixB[k*N + j];
// 						matrixC[i*N + j] = cij;
// 					}
// 				}
// 			}
//
// 			end = clock();
//
// 			cpu_time_used = ((double)(end - start));
//
// 			sum += cpu_time_used;
//
//
// 		}
//
// 	/////////////////////////////////////////////////////////////////
//
// 		/*Stop counters and store results in values */
//
// 		retval = PAPI_stop(EventSet, values);
//
// 	////////////////////////////////////////////////////////////////
//
//
// 		//Matrix Verification Using CBLAS
//
// 		//Computing Matrix Multiplication using CBLAS
// 		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC1, N);
//
// 		//Verifying the result
// 		compareMatrices(matrixC,matrixC1);
//
//
// 		printf("**************************************************************************************************************\n\n");
//
// 		printf("Avg execution time with Ctimer\n Size (NxN) %d \t\t %lf\n\n", N, (sum / 3.0));
// 		printf("GFLOPS:\t\t %lf\n\n",calculateGFLOPS(sum,size) );
// 		sum = 0;
//
// 		printf("\n\n______________________________________________________________________________________________________________\n\n");
//
// 		printf("PAPI Data/n");
//
// 		for (int ctr = 0; ctr < NUM_EVENTS; ctr++) {
//
// 			printf("/////////////////////////////////////\n");
// 			printf("%lld\n", values[ctr]);
// 			printf("/////////////////////////////////////\n");
// 		}
//
//
// 		printf("**************************************************************************************************************\n\n");
//
// 		//freeing the dynamic memory
// 		free(matrixA);
// 		free(matrixB);
// 		free(matrixC);
// 		free(matrixC1);
//
//
//
// 	}
//
// }
