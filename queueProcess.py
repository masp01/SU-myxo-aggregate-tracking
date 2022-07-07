import multiprocessing as mp
import sys
from fb_tools import fb_tools
import subprocess
from ProcessFrame import processFrame
from MakeDiagnosticVideoFrame import makeDiagnosticFrame

# Process a batch of videos, 10 at a time by
# making a textfile with the full path to an image folder on each line
# then run
#	python queueProcess.py YOUR_VIDEOS.txt

def processVideo(img_folder):
	# Prepare paths and info
	fb = fb_tools(img_folder)
	
	# Make output directories
	fb.warn_mkdir(fb.datadir)
	fb.warn_mkdir(fb.points_dir)
	fb.warn_mkdir(fb.vid_df_dir)

	# Process frames in parallel
	print('\t{}: Processing {} images'.format(fb.videoname, len(fb.img_paths)))
	frameProcesses = [mp.Process(target=processFrame, args=(img_folder, i)) for i in range(len(fb.img_paths))]
	for p in frameProcesses:
		p.start()
	for p in frameProcesses:
		p.join()
	print("{}: Processing complete".format(fb.videoname))

# Helper function for makeDiagnosticVideo()
# (passes lists of contour indices)
def intsToString(ints):
	return ','.join([str(n) for n in ints])

def makeDiagnosticVideo(img_folder):
	# Prepare paths and info
	fb = fb_tools(img_folder)
	fb.warn_mkdir(fb.viddir)

	# Get filtered contour indices and fruiting body IDs
	all_contour_indices,categorized = fb.getIDs()
	contour_indices_by_frame = [intsToString(df_slice.index) for i,df_slice in categorized.groupby('frame')]

	# Make diagnostic frames in parallel
	print('\t{}: Making diagnostic frames'.format(fb.videoname))
	dFrameProcesses = [mp.Process(target=makeDiagnosticFrame, args=(img_folder, i, contour_indices_by_frame[i])) for i in range(len(fb.img_paths))]
	for p in dFrameProcesses:
		p.start()
	for p in dFrameProcesses:
		p.join()
	print("{}: Diagnostic frames complete".format(fb.videoname))

if __name__ == "__main__":
	# Start the process queue
	img_folders = fb_tools().read_batch(sys.argv[1])
	batchSize = 10
	for i in range(int(len(img_folders)/batchSize)):
		if (i+1)*batchSize >= len(img_folders):
			batch_img_folders = img_folders[i*batchSize:]
		else:
			batch_img_folders = img_folders[i*batchSize:(i+1)*batchSize]
		processes = [mp.Process(target=processVideo, args=(img_folder,)) for img_folder in batch_img_folders]
		diagProcesses = [mp.Process(target=makeDiagnosticVideo, args=(img_folder,)) for img_folder in batch_img_folders]
		
		# Start processing the batch
		for p in processes:
			p.start()
	
		# Stop processing until the batch is complete
		for p in processes:
			p.join()

		# Start making diagnostic videos for the batch
		for p in diagProcesses:
			p.start()
	
		# Stop processing until the diagnostic videos for the batch are complete
		for p in diagProcesses:
			p.join()