// for now, this code assumes images of 1024 x 1024
// needed libraries for the project
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <cmath>
#include <cublas.h>

using namespace std;

// standard, defined variables
#define h	1024					// height of the image matrix
#define w	1024					// width of the image matrix
#define output_file "output.raw"	// raw output file

// some global variables for image matrices
unsigned char a[h * w];		// character array, representing the final array
double a_temp[h * w]; 		// temporary array, used to store numerical values

/*
	Note: we will try to first test the filtering technique without the need for color division
	Depending on the results, we'll determine whether RGB color division is really necessary.
	Here are the following necessary steps:
	1. Perform Fourier transform on the image, at a pixel-by-pixel basis
	2. Low-pass filter, averaging the current element based on neighboring elements
	3. Inverse Fourier transform, going back to the "original" image

	
*/

// first CUDA kernel, which performs Fourier transform on the image
__global__ void fourier_transform(double *in, double *out) {
	// generate block elements
	int my_x, k, t;
	my_x = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through each output element
	for (k = 0; k < h; k++) {
		// real sum: we don't worry about imaginary values in this one
		double realSum = 0;
		// iterate through each input element
		for (t = 0; t < h; t++) {
			// calculate angle and then incorporate within final solution
			double angle = 2 * M_PI * (my_x * 1024 + t) * (my_x * 1024 + k) / 1024;
			realSum += in[my_x * 1024 + t] * cos(angle);
		}
		out[my_x * 1024 + k] = realSum;
	}
}

// second kernel, which performs the low-pass filter 

// third kernel, which does the inverse Fourier transform

// the format for the executable will be the following:
// (./executable) (file_name) (number_of_threads)
int main(int argc, char **argv) {
	int i, j;
	FILE *fp;	// file container

	// variables for time measurement
	struct timespec start, stop;
	double time;

	// generating dynamic arrays, in row-major order: a[i, j] = a[i * 1024 + j]
	unsigned char *a = (unsigned char*) malloc(sizeof(unsigned char) * h * w);
	unsigned char *b = (unsigned char*) malloc(sizeof(unsigned char) * h * w);

	// temporary double variables
	double *a_temp = (double*) malloc(sizeof(double) * h * w);
	double *b_temp = (double*) malloc(sizeof(double) * h * w);

	// dynamically allocating GPU (device) matrix
	double *gpu_a, *gpu_b;
	cudaMalloc((void**) &gpu_a, sizeof(double) * h * w);
	cudaMalloc((void**) &gpu_b, sizeof(double) * h * w);

	// check whether the file can be opened or not
	if (!(fp = fopen(argv[1], "rb"))) {
		printf("Can not open the file\n");
		return 1;
	}

	// reading the contents and copying onto the temporary array
	fread(a, sizeof(unsigned char), w * h, fp);
	fclose(fp);

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			a_temp[i * h + j] = (double) a[i * h + j];
		}
	}

	// main thread code goes here
	int numThreads = atoi(argv[2]);

	// setting up the block and grid configurations
	dim3 dimBlock(numThreads);
	dim3 dimGrid(1024 / numThreads);


	// copying host onto the device
	cudaMemcpy(gpu_a, a_temp, sizeof(double) * h * w, cudaMemcpyHostToDevice);


	// measuring the start time
	if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
		perror("Clock gettime");
	}

	// the CUDA processes will go here
	// first step: Fourier transform
	fourier_transform<<<dimGrid, dimBlock>>> (gpu_a, gpu_b);
	cudaMemcpy(b_temp, gpu_b, sizeof(double) * h * w, cudaMemcpyDeviceToHost);

	// second step: going into the low-pass filter process
	// this will require having to send data from device to device

	// measure the end time here
	if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
		perror("Clock gettime");
	}
	time = (stop.tv_sec - start.tv_sec) + (double) (stop.tv_nsec - start.tv_nsec) / 1e9;


	// print out the execution time here
	printf("Execution time = %f sec\n", time);


	// creating the output file for verification
	if (!(fp = fopen(output_file, "wb"))) {
		printf("Can not open file\n");
		return 1;
	}

	// copying temporary b values onto the final b array
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			b[i * h + j] = (unsigned char) b_temp[i * h + j];
		}
	}

	// writing onto the output file and closing afterwards
	fwrite(b, sizeof(unsigned char), w * h, fp);
	fclose(fp);
	printf("\n");


	// free up all memory used
	free(a); free(b);
	free(a_temp); free(b_temp);
	cudaFree(gpu_a); cudaFree(gpu_b);
	return 0;
}