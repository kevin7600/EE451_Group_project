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
#define output_file "output.png"	// output image file

// one-dimensional mean filter, which goes through the input image and releases an output
// takes three inputs: input array, output array & size number
void average_filter_1D(double *input, double *output, int size) {
	// move the window through all elements of the input
	// assumes a window of size 5 for now.
	int i;
	for (i = 2; i < size - 2; i++) {
		output[i - 2] = (input[i - 2] + input[i - 1] + input[i] + input[i + 1] + input[i + 2]) / 5;
	}
}

// two-dimensional averga efilter
// takes in an array and gives in the output
// this function also requires the height and width of the image
void average_filter_2D(double *input, double *output, int height, int width) {
	// go through all elements of the image
	int m, n;
	for (m = 1; m < height - 1; ++m) {
		for (n = 1; n < width - 1; ++n) {
			// take the average
			output[(m - 1) * (width - 2) + n - 1] = (
				input[(m - 1) * width + n - 1] * 
				input[(m - 1) * width + n] * 
				input[(m - 1) * width + n + 1] * 
				input[m * width + n - 1] * 
				input[m * width + n] * 
				input[m * width + n + 1] * 
				input[(m + 1) * width + n - 1] * 
				input[(m + 1) * width + n] *
				input[(m + 1) * width + n + 1]) / 9;
		}
	}
}

// second kernel, which performs the low-pass filter 
// 2D array should be initialized as a result of the previous Fourier transform
// let's begin with a serial implementation, for now
void gauss_filter(double *in) {
	// first step: converting from a 1D linear array into a 2D matrix
	double arr_temp[1024][1024];
	int x, y;
	for (x = 0; x < h; x++) {
		for (y = 0; y < w; y++) {
			arr_temp[x][y] = in[x * 1024 + y];
		}
	}

	// assigning standard deviation values
	double std_dev = 1.0;
	double r = 2.0 * std_dev * std_dev;
	double s = 2.0 * std_dev * std_dev;

	// initialize the sum for normalization
	double sum = 0.0;

	// creating offset variables
	int offset_x = h / 2;
	int offset_y = w / 2;

	// loop to generate a kernel, similar in size to the original image
	for (x = -offset_x; x <= offset_x; x++) {
		for (y = -offset_y; y <= offset_y; y++) {
			double x_p = (double) x;
			double y_p = (double) y;
			r = sqrt((x_p * x_p) + (y_p * y_p));
			arr_temp[x + 1024][y + 1024] = (exp(-(r * r) / s)) / (M_PI * s);
			sum += arr_temp[x + 1024][y + 1024];
		}
	}

	// loop again to normalize the loop
	for (x = 0; x < h; x++) {
		for (y = 0; y < w; y++) {
			arr_temp[x][y] = arr_temp[x][y] / sum;
		}
	}

	// final step is to convert from 2D array to 1D row-major linear array
	for (x = 0; x < h; x++) {
		for (y = 0; y < w; y++) {
			in[x * 1024 + y] = arr_temp[x][y];
		}
	}
}

// first CUDA kernel, which performs Fourier transform on the image
// while this code loops in n^2 time, CUDA parallelism should make the program faster
__global__ void fourier_transform(double *in, double *out, int height, int width, int blockConfig) {
	// block elements
	int my_x, k, t;
	my_x = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through each element, going from frequency to time domain
	for (k = 0; k < height; k++) {
		// difference, which will be used to subtract off
		double realSum = 0;
		// iterate through the input element
		for (t = 0; t < width; t++) {
			double angle = 2 * M_PI * (my_x * height + t) * (my_x * height + k) / height;
			realSum += in[my_x * height + t] * cos(angle);
		}
		out[my_x * height + k] = realSum;
	}
}

// third kernel, which does the inverse Fourier transform
// as this only involves real numbers, it's basically an averaged Fourier transform, after all mods
__global__ void inverse_transform(double *in, double *out, int height, int width) {

	// block elements
	int my_x, k, t;
	my_x = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through each element, going from frequency to time domain
	for (k = 0; k < height; k++) {
		// difference, which will be used to subtract off
		double realSum = 0;
		// iterate through the input element
		for (t = 0; t < width; t++) {
			double angle = 2 * M_PI * (my_x * height + t) * (my_x * height + k) / height;
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
int main(int argc, char **argv) {
	// height and width variables, critical for this program
	int height, width;

	// checks for sufficient amount of arguments
	if (argc != 4) {
		printf("Invalid amount of arguments\n");
		return 1;
	}

	// checks whether a valid image mode is set
	if ((argv[3] > 2) || (argv[4] < 0)) {
		printf("Invalid image mode set\n");
		return 1;
	}

	// updates image array height and width, based on image mode selected
	// for now, square images will be tested
	// initiate 512x512 image
	if (argv[0] == 0) {
		height = 512;
		width = 512;
	}

	// initiate 1024x1024 image
	else if (argv[0] == 1) {
		height = 1024;
		width = 1024;
	}

	// initiate 2048x2048 image
	else if (argv[0] == 2) {
		height = 2048;
		width = 2048;
	}

	int i, j;	// index variables, which will iterate through the image
	FILE *fp;	// file container

	// variables for time measurement
	struct timespec start, stop;
	double time;

	// generating dynamic arrays 
	// these arrays will be allocated in row-major order: a[i, j] = a[i * 1024 + j]
	unsigned char *a = (unsigned char*) malloc(sizeof(unsigned char) * height * width);	// start array
	unsigned char *b = (unsigned char*) malloc(sizeof(unsigned char) * height * width);	// end array

	// temporary double variables
	double *a_temp = (double*) malloc(sizeof(double) * height * width);
	double *b_temp = (double*) malloc(sizeof(double) * height * width);
	double *c_temp = (double*) malloc(sizeof(double) * height * width);

	// dynamically allocating GPU (device) arrays
	// these arrays will interface directly with global functions, listed above
	double *gpu_a, *gpu_b, *gpu_c;
	cudaMalloc((void**) &gpu_a, sizeof(double) * height * width);
	cudaMalloc((void**) &gpu_b, sizeof(double) * height * width);
	cudaMalloc((void**) &gpu_c, sizeof(double) * height * width);

	// check whether the file can be opened or not
	// basically, it checks if the file is in the current directory
	if (!(fp = fopen(argv[1], "rb"))) {
		printf("Can not open the file\n");
		return 1;
	}

	// reading file contents and copy onto array a
	fread(a, sizeof(unsigned char), w * h, fp);
	fclose(fp);

	// copy to temporary array, convert unsigned char to double
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			a_temp[i * height + j] = (double) a[i * height + j];
		}
	}

	// extract an integer value from user input
	int numThreads = atoi(argv[2]);

	// setting up the block and grid configurations
	dim3 dimBlock(numThreads);
	dim3 dimGrid(1024 / numThreads);

	// copying host onto the device
	cudaMemcpy(gpu_a, a_temp, sizeof(double) * height * width, cudaMemcpyHostToDevice);

	// measuring the start time
	if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
		perror("Clock gettime");
	}

	// the CUDA processes will go here
	// first step is to divide the image into its RGB values


	// second step: perform a Fourier transform on the resulting images
	fourier_transform<<<dimGrid, dimBlock>>> (gpu_a, gpu_b, height, width, (1024 / numThreads));

	// third step: apply the mean, or average, filter on the image in the frequency domain
	// this will involve performing the filter on the second GPU

	// fourth step: compress the three arrays (RGB) back into the single array

	// final step: performing the inverse Fourier transform and obtaining the output
	//inverse_transform<<<dimGrid, dimBlock>>> (gpu_c, gpu_a);
	//cudaMemcpy(gpu_a, a_temp, sizeof(double) * h * w, cudaMemcpyDeviceToHost);

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
	return 0;
}