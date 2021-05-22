# adapted from https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_seqs.py

import os
import glob
import cv2 as cv
import numpy as np

def save_img(dname, fn, i, frame):
	cv.imwrite('{}/{}_{}_{}.png'.format(
		out_dir, os.path.basename(dname),
		os.path.basename(fn).split('.')[0], i), frame)

out_dir = 'images'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

def convert(dir):
	# print(os.getcwd())
	for dname in sorted(glob.glob(dir)):
		for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
			vidcap = cv.VideoCapture(fn)
			success, image = vidcap.read()

			# count = 0

			# while vidcap.isOpened():
			# 	success,image = vidcap.read()
			# 	if success:
			# 		save_img(dname, fn, count, image)
			# 		count += 300
			# 		vidcap.set(1, coun)

			# # TRAIL
			# # total_frames = vidcap.get(cv.CAP_PROP_FRAME_COUNT)
			# fps = vidcap.get(cv.CAP_PROP_FPS)
			# est_video_length_minutes = 1         # Round up if not sure.
			# est_tot_frames = est_video_length_minutes * 60 * fps
			# n = 3
			# frames_step = np.floor(est_tot_frames/n)
			# for i in range(n):
			# 		#here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
			# 		vidcap.set(1,i*frames_step)
			# 		success,image = vidcap.read()
			# 		frameId = vidcap.get(i*frames_step)  
			# 		#save your image
			# 		save_img(dname, fn, i*frames_step, image)
			# # TRAIL
			seconds = 15
			fps = vidcap.get(cv.CAP_PROP_FPS) # Gets the frames per second
			multiplier = fps * seconds

			while success:
					frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
					success, image = vidcap.read()

					if frameId % multiplier == 0:
							# cv.imwrite("FolderSeconds/frame%d.jpg" % frameId, image)
							save_img(dname, fn, frameId, image)


			# fps = vidcap.get(cv.CAP_PROP_FPS)
			# # print(fps) 25 fps
			# est_video_length_minutes = 1         # Round up if not sure.
			# est_tot_frames = est_video_length_minutes * 60 * fps  # Sets an upper bound # of frames in video clip

			# n = 25 * 30                             # Desired interval of frames to include
			# desired_frames = n * np.arange(est_tot_frames) 

			# #################### Initiate Process ################
			# count = 0
			# for i in desired_frames:
			# 		vidcap.set(1,i-1)                      
			# 		success,image = vidcap.read(1)         # image is an array of array of [R,G,B] values
			# 		frameId = vidcap.get(1)
			# 		# print(frameId)
			# 		save_img(dname, fn, count, image)
			# 		count+=1

			vidcap.release()
			print(fn)


convert('../caltech/train*')
convert('../caltech/test*')