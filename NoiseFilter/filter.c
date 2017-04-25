#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>



// defined variables
#define output_file "output.raw"	// output image file (bitmap)


int main(int argc, char **argv) {
	//argv[0] executable name
	//argv[1] input image name
	//argv[2] image width
	//argv[3] image height
	//argv[4]  window mode mask/window filter dimentions
	// ****** 0 = 3x3
	// ****** 1 = 5x5
	// ****** 2 = 7x7
	// ****** etc.
	//argv[5] filter type
	// ****** 0 = median
	// ****** 1 = average
	// ****** 2 = custom
	// argv[6] number of threads
	char inFile[50];
	int w;
	int h;
	int windowMode;
	int numThreads;	
	int filterType;
	if (argc==1){// ./run
		printf("\nEnter raw image file \n");
		scanf("%49s",inFile);
		printf("\nEnter image width \n");
		scanf("%u",&w);
		printf("\nEnter image height \n");
		scanf("%u",&h);
		printf("\nEnter windowMode\n");
		scanf("%u",&windowMode);
		printf ("\nEnter filter type (0=median, 1=average, 2=custom)\n");
		scanf("%u",&filterType);
		printf("\nEnter number of threads\n");
		scanf("%u",&numThreads);
	}
	else{// ./run "image.raw" "width" "height" "Window Mode" "Number of threads" 
		int x=0;
		while (argv[1][x]!='\0'){
			inFile[x]=argv[1][x];
			x++;
		}
		inFile[x]='\0';
		w=atoi(argv[2]);
		h=atoi(argv[3]);
		windowMode=atoi(argv[4]);
		filterType=atoi(argv[5]);
		numThreads=atoi(argv[6]);
	}
	int noiseColor;
	if (filterType==2){//custom filter
		printf("\nEnter a number based on the dominant color of the noise on a scale of 1 to 10 (1=white, 10=black) \n");
		scanf("%u",&noiseColor);
	}
	struct timespec start, stop; 
	
	double time;

	unsigned char* a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
	
	unsigned char a2[h*w];
	
	int z;

	FILE *fp;
	if (!(fp=fopen(inFile, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);

	fclose(fp);
	
    for (z=0;z<h*w;z++){
		a2[z]=a[z];
	}
	
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	if (filterType==0){//median
		int m;

		int windowDim=(3+(windowMode*2))*(3+(windowMode*2));
		
		#pragma omp parallel num_threads(numThreads)
		{		
			#pragma omp for schedule(static)
			
				for (m=0;m<h;++m){//m and n loop iterate through all pixels in raw image (hxw)
					int n;
					for (n=0;n<w;++n){
						int k=0;
						unsigned char* window=(unsigned char*) malloc(sizeof(unsigned char) * windowDim);
						int j;
						for (j=m-(1+windowMode);j<m+(2+windowMode);++j){//j and i loop create the window, dimentions_of_window=windowDim=(3+windowMode*2)x(3+windowMode*2) 
							if (j<0||j>h-1)continue;//deals with top and bottom edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
							int i;
							for (i=n-(1+windowMode);i<n+(2+windowMode);++i){
								if (i<0||i>w-1)continue;//deals with left and right edge cases, if part of window is off the raw image then exclude them. E.g. if 3x3 window is on a[0] (pixel 0,0) then ignore left column and top row of 3x3 window
								window[k++]=a[j*w+i];
							}
						}
						for (j=0;j<(k/2)+1;++j){//sort half+1 elements, to get middle number (median)
							int min=j;
							int l;
							for(l=j+1;l<k;++l){
								if (window[l]<window[min]){
									min=l;
								}
							}
							unsigned char temp=window[j];
							window[j]=window[min];
							window[min] = temp;
						}
						a2[(m)*(w)+(n)]=window[(k-1)/2];//set to median
						free(window);
					}
				}
		}
	}
	else if (filterType==1){//average
		int m;

		int windowDim=(3+windowMode*2)*(3+windowMode*2);
		
		#pragma omp parallel num_threads(numThreads)
		{		
			#pragma omp for schedule(static)
			
				for (m=0;m<h;++m){//m and n loop iterate through all pixels in raw image (hxw)
					int n;
					
					for (n=0;n<w;++n){
						int k=0;
						unsigned char* window=(unsigned char*) malloc(sizeof(unsigned char) * windowDim);
						int j;
						for (j=m-(1+windowMode);j<m+(2+windowMode);++j){//j and i loop create the window, dimentions_of_window=windowDim=(3+windowMode*2)x(3+windowMode*2) 
							if (j<0||j>h-1)continue;
							int i;
							for (i=n-(1+windowMode);i<n+(2+windowMode);++i){
								if (i<0||i>w-1)continue;
								window[k++]=a[j*w+i];
							}
						}
						float total=0;
						for (j=0;j<k;++j){//averages values in window
							total=total+(float)window[j];
						}
						a2[m*w+n]=total/(float)k;
						free(window);

					}
				}
		}
	}
	else if (filterType==2){//customized filter. Variation of median filter, instead of picking the middle number from sorted window, pick the number based on user input. 

		int m;

		int windowDim=(3+windowMode*2)*(3+windowMode*2);
		
		#pragma omp parallel num_threads(numThreads)
		{		
			#pragma omp for schedule(static)
			
				for (m=0;m<h;++m){
					int n;
					
					for (n=0;n<w;++n){
						int k=0;
						unsigned char* window=(unsigned char*) malloc(sizeof(unsigned char) * windowDim);
						int j;
						for (j=m-(1+windowMode);j<m+(2+windowMode);++j){
							if (j<0||j>h-1)continue;
							int i;
							for (i=n-(1+windowMode);i<n+(2+windowMode);++i){
								if (i<0||i>w-1)continue;
								window[k++]=a[j*w+i];
							}
						}
						for (j=0;j<k+1;++j){//sort all elements
							int min=j;
							int l;
							for(l=j+1;l<k;++l){
								if (window[l]<window[min]){
									min=l;
								}
							}
							unsigned char temp=window[j];
							window[j]=window[min];
							window[min] = temp;
						}
						a2[(m)*(w)+(n)]=window[(int)(noiseColor*(k-1))/10];
						free(window);					
					}
				}
		}
	}
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		// print out the execution time here
		printf("Execution time = %f sec", time);		
	
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not open file\n");
		return 1;
	}	
	fwrite(a2, sizeof(unsigned char),w*h, fp);
    fclose(fp);
	free(a);
    
    return 0;
}