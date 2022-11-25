# Transcripts Alignment - MiM Algorithm
The method allows to align the transcription of a line of text to the related words in the image of the line

## Preparation
1. Create a ```"data/lines"``` folder which contains the images of the text lines. The folder is organized into subfolders, one for each document.

2. Create the ```"data/GT"``` folder which contains the transcript txt files. The folder is organized into subfolders, one for each document.

3. Set all the input folders in the file ```configs.py```.


## Perform alignment

1. Run the ```alignment.py``` file to align and get the ```"all_align.als"``` pickle file. Within the file you can set parameters for the process.

2. You can fix the alignment algorithm outputs by running the ```"correction_tool.py"``` file
   the tool will display all the words aligned one at a time.
   With the ENTER key you can move to the next word.
   With the BACKSPACE key you go back to the previous word (of the same line)
   With the cntrl+s keys you save the state
   With the cntrl+q keys you can close the GUI
   
   To correct a segmentation fault you can use the mouse:
      with a click with the left button a new left segmentation boundary is set
      a right click sets a new right segmentation boundary
    
   finally the tool fixes the alignment file ```"all_aligns.als"```
   and generates in the ```time``` folder a file where the total time spent on the correction is reported

   The process also measures alignment performance:
   a file will be saved in the ```Performance``` folder
   where the total number of alignments and the number of alignments that did not need correction are shown


3. run the file ```crop_all_words.py``` to generate all the images of the obtained words