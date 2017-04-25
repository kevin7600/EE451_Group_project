// for now, this code assumes images are square (rows = columns)
// needed libraries for the project

// to compile: nvcc -o go main.cu -Xcompiler -fopenmp

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <cmath>
#include <cublas.h>
#include <omp.h>

// namespace directive
using namespace std;

// defined variables
#define output_file "output_image.raw"	// output image file (bitmap)

// CUDA kernel, which performs Fourier transform on the image
// while this code loops in n^2 time, CUDA parallelism should make the program faster
// only real values would be considered, as the following is extracted from the image
__global__ void fourier_transform(float *in, float *out, int height, int width, int blockConfig) {
	// block elements and function variables
	int my_x, k, t;
	my_x = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through each element, going from frequency to time domain
	for (k = 0; k < height; k++) {
		// difference, which will be used to subtract off
		float realSum = 0.0;
		// iterate through the input element
		for (t = 0; t < width; t++) {
			// calculate the angle and update the sum
			float angle = 2 * M_PI * (my_x * height + t) * (my_x * width + k) / height;
			realSum += in[my_x * height + t] * cos(angle);
		}
		// each output element will be the current sum for that index
		out[my_x * height + k] = realSum;
	}
}

// kernel, which does the inverse Fourier transform
// as this only involves real numbers, it's basically an averaged Fourier transform
__global__ void inverse_transform(float *in, float *out, int height, int width) {
	// block elements
	int my_x, k, t;
	my_x = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through each element, going from frequency to time domain
	for (k = 0; k < height; k++) {
		// difference, which will be used to subtract off
		float realSum = 0;
		// iterate through the input element
		for (t = 0; t < width; t++) {
			float angle = 2 * M_PI * (my_x * height + t) * (my_x * height + k) / height;
			realSum += in[my_x * height + t] * cos(angle);
		}
		out[my_x * height + k] = (realSum / height);
	}
}

// the format for the executable will be the following:
// argv[0] = executable name
// argv[1] = input image name
// argv[2] = number of threads for block
// argv[3] = image mode 
// ****** 0 = 256x256
// ****** 1 = 512x512
// ****** 2 = 1024x1024	
// ****** 3 = 2048x2048	
// argv[4] = convolution mask width
// argv[5] = convolution mode: slow or optimized
// argv[6] = specified OpenMP threads

int main(int argc, char **argv) {
	// height and width variables, critical for this program
	int height, width;

	// checks for sufficient amount of arguments
	if (argc != 7) {
		printf("Invalid amount of arguments\n");
		return 1;
	}

	// checks whether a valid image mode is set
	if ((atoi(argv[3]) > 3) || (atoi(argv[3]) < 0)) {
		printf("Invalid image mode set\n");
		return 1;
	}

	// updates image array height and width, based on image mode selected
	// for now, square images will be tested
	// initiate 256x256 image
	if (atoi(argv[3]) == 0) {
		height = 256;
		width = 256;
	}

	// initiate 512x512 image
	else if (atoi(argv[3]) == 1) {
		height = 512;
		width = 512;
	}

	// initiate 1024x1024 image
	else if (atoi(argv[3]) == 2) {
		height = 1024;
		width = 1024;
	}

	// initiate 2048x2048 image
	else if (atoi(argv[3]) == 3) {
		height = 2048;
		width = 2048;
	}

	int i, j;	// index variables, which will iterate through loops
	FILE *fp;	// file container

	// variables for time measurement
	struct timespec start, stop;
	float time;

	// generating dynamic arrays 
	// generating a dynamic kernel array
	float *kernel = (float*) malloc(sizeof(float) * atoi(argv[4]) * atoi(argv[4]));

	// these arrays will be allocated in row-major order: a[i, j] = a[i * 1024 + j]
	unsigned char *a = (unsigned char*) malloc(sizeof(unsigned char) * height * width);	// start array
	unsigned char *b = (unsigned char*) malloc(sizeof(unsigned char) * height * width);	// end array

	// temporary double variables
	float *a_temp = (float*) malloc(sizeof(float) * height * width);
	float *b_temp = (float*) malloc(sizeof(float) * height * width);
	float *c_temp = (float*) malloc(sizeof(float) * height * width);

	// dynamically allocating GPU (device) arrays
	// these arrays will interface directly with global functions, listed above
	float *gpu_a, *gpu_b, *gpu_c;
	cudaMalloc((void**) &gpu_a, sizeof(float) * height * width);
	cudaMalloc((void**) &gpu_b, sizeof(float) * height * width);
	cudaMalloc((void**) &gpu_c, sizeof(float) * height * width);

	// print statement, ensuring all the memory has been allocated
	printf("Dynamic memory allocation complete\n");

	// check whether the file can be opened or not
	// basically, it checks if the file is in the current directory
	if (!(fp = fopen(argv[1], "rb"))) {
		printf("Can not open the file\n");
		return 1;
	}

	// reading file contents and copy onto array a
	// this basically reads raw binary data and creates an array
	fread(a, sizeof(unsigned char), width * height, fp);
	fclose(fp);

	// copy to temporary array, convert unsigned char to double
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			a_temp[i * height + j] = (float) a[i * height + j];
			c_temp[i * height + j] = 0;
		}
	}

	// filling the kernel with contents, where the sum of everything is 1
	for (i = 0; i < atoi(argv[4]); i++) {
		for (j = 0; j < atoi(argv[4]); j++) {
			kernel[i * atoi(argv[4]) + j] = 1 / (atoi(argv[4]) * atoi(argv[4]));
		}
	}

	// extract an integer value from user input
	int numThreads = atoi(argv[2]);

	// setting up the block and grid configurations
	dim3 dimBlock(numThreads);
	dim3 dimGrid(height / numThreads);

	// copying host onto the device
	cudaMemcpy(gpu_a, a_temp, sizeof(float) * height * width, cudaMemcpyHostToDevice);

	// measuring the start time
	if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
		perror("Clock gettime");
	}

	// perform a Fourier transform on the resulting images
	// the Fourier transform will be done in paralleli, using GPU
	fourier_transform<<<dimGrid, dimBlock>>> (gpu_a, gpu_b, height, width, (height / numThreads));

	// copy Fourier results from gpu_b to b_temp
	cudaMemcpy(b_temp, gpu_b, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

	// perform a 2D image convolution on the image
	// keep in mind, the image should be in the frequency domain first
	// this method will actually be implemented using OpenMP instead
	// in case the unoptimized convolution is chosen
	if (atoi(argv[6]) == 0) {
		printf("Please input non-zero value\n");
		return 1;
	}

	if (atoi(argv[5]) == 0) {
		int i, j, m, n, mm, nn;
		int k_centerX, k_centerY;
		float sum;
		int rowIndex, colIndex;

		// find center position of the kernel
		k_centerX = atoi(argv[4]) / 2;
		k_centerY = atoi(argv[4]) / 2;

		// loop through all the rows
		// this loop would be done in parallel, using OpenMP, satisfying task parallelism
		#pragma omp parallel num_threads(atoi(argv[6]))
			#pragma omp for schedule(static) private(i) reduction(+:height)
			for (i = 0; i < height; ++i) {
				// loop through all the columns
				// this will also be in parallel, using same OpenMP methods used previously
				#pragma omp parallel num_threads(atoi(argv[6]))
					#pragma omp for schedule(static) private(j) reduction(+:width)
					for (j = 0; j < width; ++j) {
						sum = 0;
						// loop through kernel rows
						for (m = 0; m < atoi(argv[4]); ++m) {
							mm = atoi(argv[4]) - 1 - m;
							// loop through kernel columns
							for (n = 0; n < atoi(argv[4]); ++n) {
								nn = atoi(argv[4]) - 1 - n;

								rowIndex = i + m - k_centerY;
								colIndex = j + n - k_centerX;

								// ignore input samples which are out of bound
			                    if ((rowIndex >= 0) && (rowIndex < height) && (colIndex >= 0) && (colIndex < width)) {
			                        sum += b_temp[width * rowIndex + colIndex] * kernel[height * mm + nn];
			                    }
							}
						}
						c_temp[width * i + j] = (float) fabs(sum) + 0.5f;
					}
			}
	}

	// in case the optimized convolution is selected
	else if (atoi(argv[5]) == 1) {
		// first step will be mapping the image array
		// this involves mapping from a 1D linear array to a 2D mesh array
		int x, y, i, j;		// index variables

		// temporary float and kernel array, used to store the 2D mesh version
		float **temp_array;
		float **temp_out;
		float **temp_kernel;
		temp_array = (float**) malloc(sizeof(float*) * width);
		temp_out = (float**) malloc(sizeof(float*) * width);
		temp_kernel = (float**) malloc(sizeof(float*) * atoi(argv[4]));

		// mapping out 2D float array
		for (x = 0; x < height; x++) {
			temp_array[x] = (float*) malloc(sizeof(float) * height);
			temp_out[x] = (float*) malloc(sizeof(float) * height);
		}

		// mapping out 2D kernel array
		for (x = 0; x < atoi(argv[4]); x++) {
			temp_kernel[x] = (float*) malloc(sizeof(float) * atoi(argv[4]));
		}

		// mapping from 1D to 2D
		for (x = 0; x < height; x++) {
			for (y = 0; y < width; y++) {
				temp_array[x][y] = b_temp[x * height + y];
			}
		}

		// filling up the kernel mask with values, where the sum of all elements equals to 1
		for (x = 0; x < atoi(argv[4]); x++) {
			for (y = 0; y < atoi(argv[4]); y++) {
				temp_kernel[x][y] = 1 / (atoi(argv[4]) * atoi(argv[4]));
			}
		}

		// offset variable, to determine sizing
		int kern_offset = atoi(argv[4]) / 2;

		// performing the actual convolution procedure
		// each element doesn't start at 0, as it won't risk from out of bounds calculations
		// the true starting point depends on the offset, which is based on the kernel size
		#pragma omp parallel num_threads(atoi(argv[6]))
			#pragma omp for schedule(static) private(x) reduction(+:height)
			for (x = 0; x < height; ++x) {
				// loop through the width of the image
				#pragma omp parallel num_threads(atoi(argv[6]))
					#pragma omp for schedule(static) private(y) reduction(+:width)
					for (y = 0; y < width; ++y) {
						// the next series of loops will iterate through the mask
						for (i = 0; i < atoi(argv[4]); ++i) {
							// flipped kernel row element
							int flip_kern_row = atoi(argv[4]) - 1 - i;

							for (j = 0; j <= atoi(argv[4]); ++j) {
								// flipped kernel column element
								int flip_kern_col = atoi(argv[4]) - 1 - j;

								int index_row = x + (i - kern_offset);
								int index_col = y + (j - kern_offset);

								// ignore input values, which are out of bounds
								if ((index_row >= 0) && (index_row < height) && (index_col >= 0) && (index_col < width))
								{
									temp_out[x][y] += temp_array[index_row][index_col] * temp_kernel[flip_kern_row][flip_kern_col];
								}
							}
						}
					}
			}

		// printing out the output array, based on filled values
		for (x = 0; x < height; x++) {
			for (y = 0; y < width; y++) {
				c_temp[x * height + y] = temp_out[x][y];
			}
		}

		// deallocate 2D kernel array
		for (x = 0; x < atoi(argv[4]); x++) {
			free(temp_kernel[x]);
		}

		// deallocating the 2D array, once finished using it
		for (x = 0; x < height; x++) {
			free(temp_array[x]);
		}

		free(temp_kernel);
		free(temp_array);
	}

	// else, invalid convolution mode
	else {
		printf("Invalid convolution mode\n");
		return 1;
	}

	// copy the results of the b_temp array onto its respective GPU array
	cudaMemcpy(gpu_c, c_temp, sizeof(float) * height * width, cudaMemcpyHostToDevice);

	// performing the inverse Fourier transform to obtain the final image
	inverse_transform<<<dimGrid, dimBlock>>> (gpu_c, gpu_a, height, width);

	// copy the final results from gpu_a to a_temp for final processing
	cudaMemcpy(a_temp, gpu_a, sizeof(double) * height * width, cudaMemcpyDeviceToHost);
	//--------------------------------------------------------------------------------------//

	// measure the end time here
	if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
		perror("Clock gettime");
	}
	time = (stop.tv_sec - start.tv_sec) + (float) (stop.tv_nsec - start.tv_nsec) / 1e9;

	// print out relevant variables used
	printf("OpenMP threads used = %d\n", atoi(argv[6]));
	printf("Screen resolution = %dx%d\n", height, width);
	printf("CUDA threads per block = %d\n", atoi(argv[2]));
	printf("Kernel window size = %dx%d\n", atoi(argv[4]), atoi(argv[4]));
	printf("Convolution mode = %d\n", atoi(argv[5]));

	// print out the execution time here
	printf("Execution time = %f sec\n", time);

	// printing out performance throughput
	int n = height;
	printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", 2*n*n*n, time, 1/time/1e6*2*n*n*n);

	// creating the output file for verification
	if (!(fp = fopen(output_file, "wb"))) {
		printf("Can not open file\n");
		return 1;
	}

	// copying temporary b values onto the final b array
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			b[i * height + j] = (unsigned char) a_temp[i * height + j];
		}
	}

	// writing onto the output file and closing afterwards
	fwrite(b, sizeof(unsigned char), height * width, fp);
	fclose(fp);
	printf("\n");

	// free up all memory used
	free(a); 
	free(b);
	free(kernel);
	free(a_temp); 
	free(b_temp); 
	free(c_temp);
	cudaFree(gpu_a); 
	cudaFree(gpu_b); 
	cudaFree(gpu_c);

	return 0;
}