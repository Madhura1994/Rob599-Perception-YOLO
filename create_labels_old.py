import ntpath
import os
import sys

#Path to image folder and output folder
folder_name = "/home/madhura/Desktop/YOLO/rob599_dataset_deploy/trainval"
labels_name = "/home/madhura/Desktop/YOLO/rob599_dataset_deploy/labels_out"

# To allow the finding of the perception module, TODO - Find a way withot this hack
sys.path.append('../../')
sys.path.append('../')

from glob import glob
import argparse
import pandas as pd
import numpy as np
from perception.helper.data import get_all_2D_bbox, load_bbox, load_proj
from PIL import Image

def get_bbox_details(image_fn):
	"""This function will retrieve the bbox associated with a file"""
	proj_fn = image_fn.replace('_image.jpg', '_proj.bin')
	bbox_fn = image_fn.replace('_image.jpg', '_bbox.bin')
	if os.path.exists(bbox_fn):
		bbox_3d = load_bbox(bbox_fn)
		proj = load_proj(proj_fn)
		bbox_2d = get_all_2D_bbox(bbox_3d, proj, with_extras=True)
		print(bbox_2d)
		return bbox_2d
	else:
		# We dont have any bounding boxes, just put all zeros
		return [[0, 0, 0, 0, 0, False]]


## Main Starts here

#Keeping a count of total images
count = 0

#For every image in output folder
for root, dirs, files in os.walk(folder_name):
	for file in files:
		if file.endswith(".jpg"):
			file_name = os.path.join(root, file)
			im = Image.open(file_name)
			im_width = im.size[0]
			im_height = im.size[1]

			#Creating the textfile name
			dir_name = ntpath.dirname(file_name)
			base_name = ntpath.basename(file_name)
			split = ntpath.split(dir_name)
			temp_name = base_name.replace('.jpg', '.txt')
			output_name = split[1]+'_'+temp_name
			final_name = os.path.join(labels_name, output_name)
			count += 1
			print(count)

			file_handle = open(final_name, 'w')

			#Extracting bbox information
			bboxs = get_bbox_details(file_name)
			for i in range(len(bboxs)):
				temp = bboxs[i]
				if (temp[0] < 0):
					temp[0] = 0;
				if (temp[1] > im_width):
					temp[1] = im_width
				if (temp[2] < 0):
					temp[2] = 0
				if (temp[3] > im_height):
					temp[3] = im_height

				print(temp, file_name)

				width = np.abs(temp[1] - temp[0])/im_width
				height = np.abs(temp[3] - temp[2])/im_height
				x_coord = (temp[0] + temp[1])/(2*im_width)
				y_coord = (temp[2] + temp[3])/(2*im_height)
				object_class = temp[4]
				if(temp[5] == False):
					file_handle.write("%i %5.5f %5.5f %5.5f %5.5f \n" % (object_class, x_coord, y_coord, width, height))

			file_handle.close()
			



			

			
