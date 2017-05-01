"#EE451_Group_project" 

Do this:
	source /usr/usc/cuda/5.5/setup.sh

Compile code:
	nvcc -o go filter.cu

format queue.pbs:
	dos2unix queue.pbs
	
Run program:
	qsub queue.pbs
 
 
 *can get the width and height of a raw image from the respective pic in pics folder
 
 *don't forget to edit show_raw file inside FilterMedian folder to accomodate the change in width and height
 
 Reference: http://www.librow.com/articles/article-1