
//qsub -I -d. -l nodes=1:ppn=16:gpu,walltime=00:30:00
//gcc -lrt -fopenmp -o run filter.c
// ./run
// ./run "image.raw" "width" "height" "Window Mode" "Number of threads" 
// e.g. ./run raw/dog.raw 591 401 0 4
// can get the width and height of a raw image from the respective pic in pics folder
// don't forget to edit show_raw file inside FilterMedian folder to accomodate the change in width and height
 
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
	// argv[5] number of threads
	char inFile[50];
	int w;
	int h;
	int windowMode;
	int numThreads;	
	if (argc==1){// ./run
		printf("\nEnter raw image file \n");
		scanf("%49s",inFile);
		printf("\nEnter image width \n");
		scanf("%u",&w);
		printf("\nEnter image height \n");
		scanf("%u",&h);
		printf("\nEnter windowMode\n");
		scanf("%u",&windowMode);
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
		numThreads=atoi(argv[5]);
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
	
	int m;

	int windowDim=(3+windowMode*2)*(3+windowMode*2);
	
	#pragma omp parallel num_threads(numThreads)
	{		
		#pragma omp for schedule(static)
		
			for (m=0;m<h;++m){
				int n;
				
				for (n=0;n<w;++n){
					int k=0;
					unsigned char window[windowDim];
						int j;
						for (j=m-(1+windowMode);j<m+(2+windowMode);++j){
							if (j<0||j>h-1)continue;
							int i;
							for (i=n-(1+windowMode);i<n+(2+windowMode);++i){
								if (i<0||i>w-1)continue;
								window[k++]=a[j*w+i];
							}
						}
						for (j=0;j<(k/2)+1;++j){
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
					a2[(m)*(w)+(n)]=window[k/2];
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
    
    return 0;
}