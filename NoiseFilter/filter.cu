#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define n 512 //image can now only be square images (l=w)
#define block_size 32//length of block=32 here
#define windowMode 1// 0=3x3, 1=5x5, 7=7x7, etc.
#define inFile "raw/Pepper.raw"
#define filterType 2 // 0=median, 1=mean, 2=custom filter
#define output_file "output.raw"	// output image file (bitmap)
#define noiseColor 9//for custom filter

__global__ void median_filter(unsigned char* image){
	int row=threadIdx.y;//row in the block
	int col=threadIdx.x;//col in the block
	int my_x=blockIdx.x*blockDim.x+threadIdx.x;//row in the grid
	int my_y=blockIdx.y*blockDim.y+threadIdx.y;//col in the grid
	//first create shared mem
	__shared__ unsigned char cache[block_size][block_size];
	cache[threadIdx.y][threadIdx.x]=image[my_y*n+my_x];
	syncthreads();
	//do rest the same, except: 1. access from cache rather than directly from the array
						//		2. iterate through a block rather than whole image
	
	int j,i;//to walk through window
	const int windowDim=(3+(windowMode*2))*(3+(windowMode*2));
	int windowIndex=0;
	unsigned char window[windowDim];
	for (j=row-(1+windowMode);j<row+(2+windowMode);++j){//j and i loop create the window, dimentions_of_window=windowDim=(3+windowMode*2)x(3+windowMode*2) 
		if (j<0||j>block_size-1)continue;//deals with top and bottom edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
		for (i=col-(1+windowMode);i<col+(2+windowMode);++i){
			if (i<0||i>block_size-1)continue;//deals with left and right edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
			window[windowIndex++]=cache[j][i];
		}
	}		
	for (j=0;j<(windowIndex/2)+1;++j){//sort half+1 elements, to get middle number (median)
		int min=j;
		int l;
		for(l=j+1;l<windowIndex;++l){
			if (window[l]<window[min]){
				min=l;
			}
		}
		unsigned char temp=window[j];
		window[j]=window[min];
		window[min] = temp;
	}
	image[(my_y*n)+my_x]=window[(windowIndex-1)/2];//set to median
	
}

__global__ void mean_filter(unsigned char* image){
	int row=threadIdx.y;//row in the block
	int col=threadIdx.x;//col in the block
	int my_x=blockIdx.x*blockDim.x+threadIdx.x;//row in the grid
	int my_y=blockIdx.y*blockDim.y+threadIdx.y;//col in the grid
	//first create shared mem
	__shared__ unsigned char cache[block_size][block_size];
	cache[threadIdx.y][threadIdx.x]=image[my_y*n+my_x];
	syncthreads();
	//do rest the same, except: 1. access from cache rather than directly from the array
						//		2. iterate through a block rather than whole image
	
	int j,i;//to walk through window
	const int windowDim=(3+(windowMode*2))*(3+(windowMode*2));
	int windowIndex=0;
	unsigned char window[windowDim];
	for (j=row-(1+windowMode);j<row+(2+windowMode);++j){//j and i loop create the window, dimentions_of_window=windowDim=(3+windowMode*2)x(3+windowMode*2) 
		if (j<0||j>block_size-1)continue;//deals with top and bottom edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
		for (i=col-(1+windowMode);i<col+(2+windowMode);++i){
			if (i<0||i>block_size-1)continue;//deals with left and right edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
			window[windowIndex++]=cache[j][i];
		}
	}
	float total=0;
	for (j=0;j<windowIndex;++j){//averages values in window
		total=total+(float)window[j];
	}
	image[(my_y*n)+my_x]=total/(float)windowIndex;
}
__global__ void custom_filter(unsigned char* image){
	int row=threadIdx.y;//row in the block
	int col=threadIdx.x;//col in the block
	int my_x=blockIdx.x*blockDim.x+threadIdx.x;//row in the grid
	int my_y=blockIdx.y*blockDim.y+threadIdx.y;//col in the grid
	//first create shared mem
	__shared__ unsigned char cache[block_size][block_size];
	cache[threadIdx.y][threadIdx.x]=image[my_y*n+my_x];
	syncthreads();
	//do rest the same, except: 1. access from cache rather than directly from the array
						//		2. iterate through a block rather than whole image
	
	int j,i;//to walk through window
	const int windowDim=(3+(windowMode*2))*(3+(windowMode*2));
	int windowIndex=0;
	unsigned char window[windowDim];
	for (j=row-(1+windowMode);j<row+(2+windowMode);++j){//j and i loop create the window, dimentions_of_window=windowDim=(3+windowMode*2)x(3+windowMode*2) 
		if (j<0||j>block_size-1)continue;//deals with top and bottom edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
		for (i=col-(1+windowMode);i<col+(2+windowMode);++i){
			if (i<0||i>block_size-1)continue;//deals with left and right edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
			window[windowIndex++]=cache[j][i];
		}
	}
	for (j=0;j<windowIndex+1;++j){//sort all elements
		int min=j;
		int l;
		for(l=j+1;l<windowIndex;++l){
			if (window[l]<window[min]){
				min=l;
			}
		}
		unsigned char temp=window[j];
		window[j]=window[min];
		window[min] = temp;
	}
	image[(my_y*n)+my_x]=window[(int)(noiseColor*(windowIndex-1))/10];
}
int main(int argc, char **argv) {	
	struct timespec start, stop; 
	double time;

	unsigned char* image = (unsigned char*) malloc (sizeof(unsigned char)*n*n);
	unsigned char* gpu_a;
	cudaMalloc((void**)&gpu_a, sizeof(unsigned char)*n*n); 
	
	FILE *fp;
	if (!(fp=fopen(inFile, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(image, sizeof(unsigned char), n*n, fp);
	fclose(fp);
	cudaMemcpy(gpu_a, image, sizeof(unsigned char)*n*n, cudaMemcpyHostToDevice);
	
	
	
	//cuda stuff below here
	dim3 dimGrid(n/block_size,n/block_size);
	dim3 dimBlock(block_size,block_size);
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	if (filterType==0){//median
		printf("in median filter!\n");
		median_filter<<<dimGrid,dimBlock>>>(gpu_a);
	}
	else if (filterType==1){//average
			printf("in average filter!\n");

		mean_filter<<<dimGrid,dimBlock>>>(gpu_a);
	}
	else if (filterType==2){//customized filter. Variation of median filter, instead of picking the middle number from sorted window, pick the number based on user input. 
				printf("in custom filter!\n");

		custom_filter<<<dimGrid,dimBlock>>>(gpu_a);
	}
	//end cuda stuff
	
	cudaMemcpy(image, gpu_a, sizeof(unsigned char)*n*n, cudaMemcpyDeviceToHost);
	
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		// print out the execution time here
		printf("Execution time = %f sec", time);		
	
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not open file\n");
		return 1;
	}	
	fwrite(image, sizeof(unsigned char),n*n, fp);
    fclose(fp);
	free(image);
	cudaFree(gpu_a);
    
    return 0;
}