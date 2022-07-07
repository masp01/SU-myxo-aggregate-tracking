import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as PathEffects
import cv2
import pandas as pd
import sys
from fb_tools import fb_tools
matplotlib.use('Agg') # Images are processed without opening a display window

def stringToInts(string):
	intStrings = string.split(',')
	return [int(n) for n in intStrings]

def makeDiagnosticFrame(img_folder, framenum, contour_indices):
	# Prepare paths and info
	fb = fb_tools(img_folder, framenum)

	# Open image
	#
	print('Opening image')
	#
	frame = cv2.imread(fb.img_path, 0) # 0 opens image in grayscale
	
	# Load contour points data
	#
	print('Loading contour points data')
	contours = np.load(fb.points_path, allow_pickle=True)

	contours = contours[stringToInts(contour_indices)]

	# Draw contours onto frame and display image
	output = cv2.drawContours(frame, contours, -1, 255, 2)
	aspect = np.shape(frame)[0] / np.shape(frame)[1]
	plt.subplots(figsize=(10,10*aspect))
	plt.imshow(output, cmap='gray')

	# Draw fruiting body IDs onto image and save final image
	tracked = pd.read_csv(fb.full_vid_df_path, sep='\t', index_col=0)
	frame_info = tracked[tracked.frame.values == framenum]
	for fb_i,row in frame_info.groupby('particle'):
		cx = row.x.values[0]
		cy = row.y.values[0]
		txt = plt.text(cx, cy, fb_i, color='white')
		txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
	plt.savefig(fb.vid_frame)
	plt.close()
	#
	print('\t###Saved image at ' + fb.vid_frame)
	#

if __name__ == "__main__":
    print('Drawing contours and IDs on frame ' + sys.argv[2])
    makeDiagnosticFrame(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    print('\tFrame ' + sys.argv[2] + ' complete.')