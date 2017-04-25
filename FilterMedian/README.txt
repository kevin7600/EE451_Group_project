"#EE451_Group_project" 

Reserve some nodes:
	qsub -I -d. -l nodes=1:ppn=16:gpu,walltime=00:30:00
Compile code:
	gcc -lrt -fopenmp -o run filter.c
Run program:
 ./run
 ./run "image.raw" "width" "height" "Window Mode" "Number of threads" 
 e.g. ./run raw/dog.raw 591 401 0 4
 
 
 *can get the width and height of a raw image from the respective pic in pics folder
 
 *don't forget to edit show_raw file inside FilterMedian folder to accomodate the change in width and height