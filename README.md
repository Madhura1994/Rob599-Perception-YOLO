# Rob 599 Perception YOLO

## Dependencies
0. Working on Ubuntu 16.04 OS
1. Ensure you have python 3 & pip3 installed
2. `pip3 install -r requirements.txt`
3. Install darknet (run the makefile inside darknet folder)

## Data Extract Module
1. 'extract_images' - A bash script to extract all the images in test/ trainval dataset and place in a folder. 
					  Modify: Path to the dataset (line 25)
2. 'list_images' - A bash script to save paths to all images in a datset in a text file. 
3. 'test.txt' - An example of output file generated by bash script 'list_image'
4. 'create_labels.py' - Python script to generate 2d bounding box labels in YOLO format (combined 5 classes)
					  Modify: Path to dataset and labels folder (line 6 & 7)
5. 'create_labels_old.py' - Python script to generate 2D bounding box labels in YOLO format (original 23 classes)
						  Modify: Path to dataset and labels folder (line 6 & 7)
6. 'data.py' - Numerous helper functions to convert 3D bounding box labels to 2D

## Darknet Module
1. Contails all the .c files for creating, training and testing the YOLO network. 

## Testing the YOLO Results
1. Run extract_images and list_images to generate test data in YOLO format
2. /darknet/cfg/rob599.data - Edit the paths to test.txt file created in previous step
3. ./darknet detector test cfg/rob599.data cfg/yolo_rob599.cfg yolo_rob599_final.weights -thresh 0.2




