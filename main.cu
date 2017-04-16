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

// for now

using namespace std;

// standard, defined variables
#define output_file "output_image.raw"	// output image file

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
			float angle = 2 * M_PI * (my_x * height + t) * (my_x * height + k) / height;
			realSum += in[my_x * height + t] * cos(angle);
		}
		// each output element will be the current sum for that index
		out[my_x * height + k] = realSum;
	}
}

// another function, which is responsible for performing 2D convolution on the image
// to continue with the process, the image should be in the frequency domain
void image_convolution(float *input, float *output, int height, int width, int kern_size) {
	// first step will be mapping the image array
	// this involves mapping from a 1D linear array to a 2D mesh array
	int x, y, i, j;		// index variables

	// temporary float and kernel array, used to store the 2D mesh version
	float **temp_array;
	float **temp_kernel;
	temp_array = (float**) malloc(sizeof(float*) * width);
	temp_kernel = (float**) malloc(sizeof(float*) * kern_size);

	// mapping out 2D float array
	for (x = 0; x < height; x++) {
		temp_array[x] = (float*) malloc(sizeof(float) * height);
	}

	// mapping out 2D kernel array
	for (x = 0; x < kern_size; x++) {
		temp_kernel[x] = (float*) malloc(sizeof(float) * kern_size);
	}

	// mapping from 1D to 2D
	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			temp_array[x][y] = input[x * height + y];
		}
	}

	// filling up the kernel mask with values, where the sum of all elements equals to 1
	for (x = 0; x < kern_size; x++) {
		for (y = 0; y < kern_size; y++) {
			temp_kernel[x][y] = 1 / (kern_size * kern_size);
		}
	}

	// offset variable, to determine sizing
	int kern_offset = kern_size / 2;

	// performing the actual convolution procedure
	// each element doesn't start at 0, as it won't risk from out of bounds calculations
	// the true starting point depends on the offset, which is based on the kernel size
	for (x = kern_offset; x < height - kern_offset; x++) {
		for (y = kern_offset; y < width - kern_offset; y++) {
			// initialize accumulated value (unweighted sum) and weighted sum
			float addValue = 0;
			float weightedSum = 0;

			// the next series of loops will iterate through the mask
			for (i = -kern_offset; i <= kern_offset; i++) {
				for (j = -kern_offset; j <= kern_offset; j++) {
					// updating weighted sum values
					float posValue = temp_array[x + i][y + j];
					addValue += posValue * temp_kernel[kern_offset + i][kern_offset + j];
					weightedSum += temp_kernel[kern_offset + i][kern_offset + j];
				}
			}

			// updating each element of the larger array
			temp_array[x][y] = (addValue / weightedSum);
		}
	}

	// printing out the output array, based on filled values
	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			output[x * height + y] = temp_array[x][y];
		}
	}

	// deallocate 2D kernel array
	for (x = 0; x < kern_size; x++) {
		free(temp_kernel[x]);
	}

	// deallocating the 2D array, once finished using it
	for (x = 0; x < height; x++) {
		free(temp_array[x]);
	}

	free(temp_kernel);
	free(temp_array);
}

// ------------------- necessary functions for convolution filter -------------------//
// image reflection function
// any pixel lying outside of the image would be reflected back to the image
int reflect(int width, int pixel) {
	if (pixel < 0) {
		return -pixel - 1;
	}
	if (pixel >= width) {
		return 2 * width - pixel - 1;
	}
	return pixel;
}

// circular indexing function
// any coordinates that exceeds bounds would be wrapped to the opposite side
int circular(int width, int pixel) {
	if (pixel < 0) {
		return pixel + width;
	}
	if (pixel >= width) {
		return pixel - width;
	}
	return pixel;
}


// third kernel, which does the inverse Fourier transform
// as this only involves real numbers, it's basically an averaged Fourier transform, after all mods
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
// ****** 0 = 512x512	---> block image
// ****** 1 = 1024x1024	---> 720p image
// ****** 2 = 2048x2048	---> 900p image
// argv[4] = convolution mask width
int main(int argc, char **argv) {
	// height and width variables, critical for this program
	int height, width;

	// checks for sufficient amount of arguments
	if (argc != 5) {
		printf("Invalid amount of arguments\n");
		return 1;
	}

	// checks whether a valid image mode is set
	if ((atoi(argv[3]) > 2) || (atoi(argv[3]) < 0)) {
		printf("Invalid image mode set\n");
		return 1;
	}

	// updates image array height and width, based on image mode selected
	// for now, square images will be tested
	// initiate 512x512 image
	if (atoi(argv[0]) == 0) {
		height = 512;
		width = 512;
	}

	// initiate 1024x1024 image
	else if (atoi(argv[0]) == 1) {
		height = 1024;
		width = 1024;
	}

	// initiate 2048x2048 image
	else if (atoi(argv[0]) == 2) {
		height = 2048;
		width = 2048;
	}

	int i, j;	// index variables, which will iterate through loops
	FILE *fp;	// file container

	// variables for time measurement
	struct timespec start, stop;
	float time;

	// generating dynamic arrays 
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
	fread(a, sizeof(unsigned char), width * height, fp);
	fclose(fp);

	// copy to temporary array, convert unsigned char to double
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			a_temp[i * height + j] = (float) a[i * height + j];
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

	// --------------------- the CUDA processes will go here --------------------------------//
	// perform a Fourier transform on the resulting images
	fourier_transform<<<dimGrid, dimBlock>>> (gpu_a, gpu_b, height, width, (height / numThreads));
	cudaMemcpy(b_temp, gpu_b, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

	// perform a 2D image convolution on the image
	// keep in mind, the image should be in the frequency domain first
	image_convolution(b_temp, c_temp, height, width, atoi(argv[4]));

	// copy the results of the c_temp array onto its respective GPU array
	cudaMemcpy(c_temp, gpu_c, sizeof(float) * height * width, cudaMemcpyHostToDevice);

	// performing the inverse Fourier transform to obtain the final image
	inverse_transform<<<dimGrid, dimBlock>>> (gpu_c, gpu_a, height, width);
	cudaMemcpy(gpu_a, a_temp, sizeof(double) * height * width, cudaMemcpyDeviceToHost);
	//--------------------------------------------------------------------------------------//

	// measure the end time here
	if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
		perror("Clock gettime");
	}
	time = (stop.tv_sec - start.tv_sec) + (float) (stop.tv_nsec - start.tv_nsec) / 1e9;


	// print out the execution time here
	printf("Execution time = %f sec\n", time);


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
	free(a_temp); 
	free(b_temp); 
	free(c_temp);
	cudaFree(gpu_a); 
	cudaFree(gpu_b); 
	cudaFree(gpu_c);

	// print statement, ensuring all memory is freed
	printf("Memory dump complete\n");

	return 0;
}