// for now, this code assumes images are square (rows = columns)
// needed libraries for the project
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <cmath>
#include <cublas.h>

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

// slow convolution function, which doesn't include any optimization
// this is essentially a general convolution technique, which performs in 0(n^2)
void convolution_slow(float *input, float *output, int height, int width, float *kernel, int kern_size) {
	int i, j, m, n, mm, nn;
	int kCenterX, kCenterY;
	float sum;
	int rowIndex, colIndex;

	// find center position of the kernel
	kCenterX = kern_size / 2;
	kCenterY = kern_size / 2;

	// loop through all the rows
	for (i = 0; i < height; ++i) {
		// loop through all the columns
		for (j = 0; j < width; ++j) {
			sum = 0;
			// loop through kernel rows
			for (m = 0; m < kern_size; ++m) {
				mm = kern_size - 1 - m;
				// loop through kernel columns
				for (n = 0; n < kern_size; ++n) {
					nn = kern_size - 1 - n;

					rowIndex = i + m - kCenterY;
					colIndex = j + n - kCenterX;

					// ignore input samples which are out of bound
                    if ((rowIndex >= 0) && (rowIndex < height) && (colIndex >= 0) && (colIndex < width)) {
                        sum += input[width * rowIndex + colIndex] * kernel[height * mm + nn];
                    }
				}
			}
			output[width * i + j] = (float) fabs(sum) + 0.5f;
		}
	}
}

// an optimized version of 2D convolution
// for the most case, this assumes the kernel is center originated
// examples includes kernel sizes of 3x3, 5x5, 7x7, etc.
void convolution_2D(float *input, float *output, int height, int width, float *kernel, int kern_size) {
	int i, j, m ,n;
	float *input_point1, *input_point2, *out_point, *kPtr;
	int kCenterX, kCenterY;
	int row_min, row_max;		// check input array boundaries
	int col_min, col_max;		// same as the two ints above

	// find the center position of the kernel
	// this will be half of the original kernel size
	kCenterX = kern_size >> 1;
	kCenterY = kern_size >> 1;

	// initialize working pointers
	input_point1 = input_point2 = &input[height * kCenterY + kCenterX];	 	// note shifted element
	out_point = output;
	kPtr = kernel;

	// begin the convolution procedure
	for (i = 0; i < height; ++i) {
		// compute convolution range, where the current row should be in between
		row_max = i + kCenterY;
		row_min = i - height + kCenterY;

		for (j = 0; j < width; ++j) {
			// compute convolution range, where the current column is in between
			col_max = j + kCenterX;
			col_min = j - width + kCenterX;

			*out_point = 0;		// set output pointer to 0 before summing

			// flip the kernel and then traverse through kernel values
			// multiply each kernel element along with input data below it
			for (m = 0; m < kern_size; ++m) {
				// check for out of bounds values
				if ((m <= row_max) && (m > row_min)) {
					for (n = 0; n < kern_size; ++n) {
						// check kernel boundaries again
						if ((n <= col_max) && (n > col_min)) {
							*out_point += *(input_point1 - n) * *kPtr;
						}
						++kPtr;		// go to the next kernel
					}
				}
				else {
					kPtr += kern_size;		// move to next row if out of bounds
				}
				input_point1 -= kern_size;
			}

			// complete finishing touches
			// pointers can be modified as they go directly to variable addresses
			kPtr = kernel;						// reset the kernel
			input_point1 = ++input_point2;		// go to the next input
			++out_point;						// go to the next output
		}
	}
}

// final convolution method, which will be a fast method
// this function essentially divides the input into sectors
// in this case, the input will be partitioned into different parts, so no need to check boundaries in all

// NOTE: this is a limited case, which will work on a special case
// as there's 9 partitions, this scenario only works for 3x3 kernels
void convolution_fast(float *input, float *output, int height, int width, float *kernel, int kern_size) {
	int i, j, m, n, x, y, t;
	float **in_point, *out_point, *point;
	int kCenterX, kCenterY;
	int row_end, col_end;		// end indices for section divider
	float sum;					// accumulation buffer
	int k, kSize;

	// determine center position of the kernel
	kCenterX = kern_size >> 1;
	kCenterY = kern_size >> 1;
	kSize = kern_size * kern_size;		// total kernel size

	// allocate multi-cursor memory
	in_point = new float*[kSize];

	// set initial position of the multi-cursor
	// the position will be a swap, instead of a kernel
	point = input + (height * kCenterY + kCenterX);		// first cursor shifted
	for (m = 0, t = 0; m < kern_size; ++m) {
		for (n = 0; n < kern_size; ++n, ++t) {
			in_point[t] = point - n;
		}
		point -= width;
	}

	// initialize working pointers
	out_point = output;

	row_end = height - kCenterY;		// bottom row partition
	col_end = width - kCenterX;			// bottom column partition

	// perform convolution on rows from index 0 to (kCenterY - 1)
	y = kCenterY;
	for (i = 0; i < kCenterY; ++i) {
		// first partition
		x = kCenterX;
		// this goes on rows from index 0 to (kCenterX - 1)
		for (j = 0; j < kCenterX; ++j) {
			sum = 0;
			t = 0;
			for (m = 0; m <= y; ++m) {
				for (n = 0; n <= x; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
				t += (kern_size - x - 1);
			}

			// storing the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next
				++in_point[k];
			}
		}

		// second partition
		for (j = kCenterX; j < col_end; ++j) {
			sum = 0;
			t = 0;
			for (m = 0; m <= y; ++m) {
				for (n = 0; n < kern_size; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}

		// third partition
		x = 1;
		for (j = col_end; j < width; ++j) {
			sum = 0;
			t = x;
			for (m = 0; m <= y; ++m) {
				for (n = 0; n < kern_size; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
				// move on to the next row
				t += x;
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}
		++y;
	}

	// convolve rows from kCenterY to (rows - kCenterY - 1)
	for (i = kCenterY; i < row_end; ++i) {
		// fourth partition
		x = kCenterX;
		for (j = 0; j < kCenterX; ++j) {
			sum = 0;
			t = 0;
			for (m = 0; m < kern_size; ++m) {
				for (n = 0; n <= x; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
				t += (kern_size - x - 1);
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}

		// fifth partition
		for (j = kCenterX; j < col_end; ++j) {
			sum = 0;
			t = 0;
			for (m = 0; m < kern_size; ++m) {
				for (n = 0; n < kern_size; ++n) {
					sum += *in_point[t] * kernel[t];
					// all cursors would be used to convolve
					// in this case, cursors would move by default
					++in_point[t];
					++t;
				}
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
		}

		// sixth partition
		x = 1;
		for (j = col_end; j < width; ++j) {
			sum = 0;
			t = x;
			for (m = 0; m < kern_size; ++m) {
				for (n = x; n < kern_size; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
				t += x;
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}
	}

	// convolve rows, those not included in the previous partitioning
	y = 1;
	for (i = row_end; i < height; ++i) {
		// seventh partition
		x = kCenterX;
		for (j = 0; j < kCenterX; ++j) {
			sum = 0;
			t = kern_size * y;

			for (m = y; m < kern_size; ++m) {
				for (n = 0; n <= x; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
				t += (kern_size - x - 1);
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}

		// eight partition
		for (j = kCenterX; j < col_end; ++j) {
			sum = 0;
			t = kern_size * y;
			for (m = y; m < kern_size; ++m) {
				for (n = 0; n < kern_size; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}

		// ninth partition
		x = 1;
		for (j = col_end; j < width; ++j) {
			sum = 0;
			t = kern_size * y + x;
			for (m = y; m < kern_size; ++m) {
				for (n = x; n < kern_size; ++n) {
					sum += *in_point[t] * kernel[t];
					++t;
				}
				t += x;
			}

			// store the output
			*out_point = (float) fabs(sum) + 0.5f;
			++out_point;
			++x;
			for (k = 0; k < kSize; ++k) {
				// shift cursors to the next 
				++in_point[k];
			}
		}

		++y;		// starting row index increased
	}
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
	if ((atoi(argv[3]) > 3) || (atoi(argv[3]) < 0)) {
		printf("Invalid image mode set\n");
		return 1;
	}

	// updates image array height and width, based on image mode selected
	// for now, square images will be tested
	// initiate 512x512 image
	if (atoi(argv[3]) == 0) {
		height = 512;
		width = 512;
	}

	// initiate 1024x1024 image
	else if (atoi(argv[3]) == 1) {
		height = 1024;
		width = 1024;
	}

	// initiate 2048x2048 image
	else if (atoi(argv[3]) == 2) {
		height = 2048;
		width = 2048;
	}

	// initiate 625x625 image image
	else if (atoi(argv[3]) == 3) {
		height = 625;
		width = 625;
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
	fread(a, sizeof(unsigned char), width * height, fp);
	fclose(fp);

	// copy to temporary array, convert unsigned char to double
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			a_temp[i * height + j] = (float) a[i * height + j];
			c_temp[i * height + j] = 50.0;
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

	// --------------------- the CUDA processes will go here --------------------------------//
	// perform a Fourier transform on the resulting images
	fourier_transform<<<dimGrid, dimBlock>>> (gpu_a, gpu_b, height, width, (height / numThreads));

	// copy Fourier results from gpu_b to b_temp
	cudaMemcpy(b_temp, gpu_b, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

	// perform a 2D image convolution on the image
	// keep in mind, the image should be in the frequency domain first
	convolution_slow(b_temp, c_temp, height, width, kernel, atoi(argv[4]));

	// copy the results of the c_temp array onto its respective GPU array
	cudaMemcpy(c_temp, gpu_c, sizeof(float) * height * width, cudaMemcpyHostToDevice);

	// performing the inverse Fourier transform to obtain the final image
	inverse_transform<<<dimGrid, dimBlock>>> (gpu_c, gpu_a, height, width);

	// copy the final results from gpu_a to a_temp for final processing
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
	free(kernel);
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