import numpy as np
import os
import re
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter

#TODO: Add a function that overlays contours with fruiting body fate

class fb_tools:
	def __init__(self, identifier=None, framenum=0, data_parent='/home/user/FruitingBodyTracking/', cat_df_parent='/home/user/Dataframes_catFB/', img_parent='/mnt/SeagateBackup_USB/Mutant_Movies/'):
		self.identifier = identifier
		self.framenum = framenum
		self.data_parent = self.ensure_final_slash(data_parent)
		self.cat_df_parent = self.ensure_final_slash(cat_df_parent)
		self.img_parent = self.ensure_final_slash(img_parent)
		
		# Load the video registry
		try:
			self.registry = pd.read_csv('Registry.txt', sep='\t', index_col=0)
		except FileNotFoundError:
			self.registry = None

		# Handle identifier as either the img_folder, the videoname, or the registry index
		# setting the img_folder, a nice, verbose videoname, and the registry index
		try:
			int(self.identifier)
			isIntLike = True
		except:
			isIntLike = False
		if isinstance(self.identifier, str):
			if '/' in self.identifier:
				# Identifier is an img_folder
				self.img_folder = self.identifier
				# Ensure img_folder doesn't end with '/' before converting to videoname
				if self.img_folder[-1] == '/':
					self.img_folder = self.img_folder[:-1]
				self.videoname = self.make_videoname(self.img_folder)
			elif '--' in self.identifier:
				# Identifier is a videoname
				if self.registry is not None:
					self.identifier = self.identifier.replace('.txt', '') # Remove .txt extension in case a DF filename is being used
					matches = self.registry[self.registry.videoname == self.identifier.replace('.txt', '')]
					if len(matches) == 1:
						self.img_folder = matches.img_folder.values[0]
						self.videoname = self.identifier
					else:
						self.identifier = None
				else:
					self.identifier = None
			else:
				self.identifier = None
		elif isIntLike:
			# Identifier is a registry index number
			if self.registry is not None:
				if self.identifier in self.registry.index:
					self.img_folder = self.registry.loc[self.identifier].img_folder
					self.videoname = self.registry.loc[self.identifier].videoname
				else:
					self.identifier = None
			else:
				self.identifier = None
		else:
			self.identifier = None

		if self.identifier is None:
			# Pick a default processed video
			self.img_folder = self.img_parent + 'ABC Transporters/MXAN_1155/MXAN_1155-mic-6-10-8-12'
			self.videoname = self.make_videoname(self.img_folder)
			self.index = 103
		elif self.registry is not None:
			# Determine the video index if the registry is available
			registry_slice = self.registry[self.registry.videoname == self.videoname]
			if len(registry_slice) > 0:
				self.index = self.registry[self.registry.videoname == self.videoname].index.values[0]
			else:
				self.index = None
		else:
			self.index = None

		# Prepare input and output directories
		self.datadir = self.data_parent + self.videoname + '/'
		
		self.vid_df_dir = self.datadir + 'UnfilteredDataFrames/'
		
		self.vid_df_path = self.vid_df_dir + self.videoname + '_{:04d}.txt'.format(self.framenum)

		self.viddir = self.datadir + 'VideoFrames/'

		self.cat_viddir = self.datadir + 'CatVideoFrames/'

		#self.vid_path = self.viddir + self.videoname + '_{:04d}.png'
		self.vid_frame = self.viddir + self.videoname + '_{:04d}.png'.format(self.framenum)

		self.cat_vid_frame = self.cat_viddir + self.videoname + '_{:04d}.png'.format(self.framenum)

		self.points_dir = self.datadir + 'ContourPoints/'

		self.points_path = self.points_dir + self.videoname + '_{:04d}.npy'.format(self.framenum)

		self.full_vid_df_path = self.datadir + self.videoname + '.txt'

		self.cat_df_path = self.cat_df_parent + self.videoname + '.txt'

		# Prepare paths to all raw images
		try:
			self.img_paths = self.get_img_paths(self.img_folder)
			self.img_path = self.img_paths[framenum]
		except FileNotFoundError:
			self.img_paths = None
			self.img_path = None

		# Load dataframe
		try:
			self.df = pd.read_csv(self.cat_df_path, sep='\t')
		except FileNotFoundError:
			self.df = None

		# Load registry error code information
		self.errorCodes = {1:'Processed with fatal errors', 2:'Magnification not 4X', 4:'Video duration not 24hrs', 8:'AG strain'}

	def checkForError(self, val, errCode):
		if errCode not in self.errorCodes.keys():
			codeList = [str(n) for n in list(self.errorCodes.keys())]
			print('Error codes are only {}'.format(', '.join(codeList)))
			return
		dropSmallerCodes = val - val%errCode
		return (dropSmallerCodes/errCode)%2 == 1

	def errorSummary(self, registry_sample=None, print_summary=True):
		errorString = ''
		# Default to reporting on errors in the whole registry
		if registry_sample is None:
			if self.registry is None:
				print('Registry not found.')
				return errorString
			else:
				registry_sample = self.registry

		totalVideos = len(registry_sample)
		totalNoErrors = len(registry_sample[(registry_sample.error == 0) & registry_sample.processed])
		totalUnprocessed = len(registry_sample[registry_sample.processed == False])
		
		errorString += '{} total videos\n\n'.format(totalVideos)

		errorString += 'Error totals:\n'
		errorString += '{}\t: No errors\n'.format(totalNoErrors)
		errorString += '{}\t: Not yet processed\n'.format(totalUnprocessed)
		sumTotal = totalNoErrors + totalUnprocessed
		for ec in self.errorCodes.keys():
			total = sum([self.checkForError(val, ec) for val in registry_sample.error])
			errorString += '{}\t: {}\n'.format(total, self.errorCodes[ec])
			sumTotal += total

		if sumTotal > totalVideos:
			errorString += '\nIntersectional error codes:\n'
			errorCounts = registry_sample.error.value_counts()
			foundCodes = [[n for n in self.errorCodes.keys() if self.checkForError(ec, n)] for ec in errorCounts.index.values]
			errorDescs = ['; '.join([self.errorCodes[n] for n in fc]) for fc in foundCodes]
			df_error = pd.DataFrame(errorCounts.rename('count'))
			df_error['desc'] = errorDescs
			df_error.loc[0, 'desc'] = 'No error or unprocessed'
			errorString += str(df_error)

		if print_summary:
			print(errorString)
		else:
			return errorString

	def ensure_final_slash(self, dir_path):
		if dir_path[-1] != '/':
			dir_path = dir_path + '/'
		return dir_path

	def make_videoname(self, img_folder):
		# Prepare a nice verbose name for each video
		videoname = img_folder.replace('H:\\', '')
		videoname = videoname.replace('\\', '--')
		videoname = videoname.replace('/media/user/Seagate Backup Plus Drive/Mutant_Movies/', '')
		videoname = videoname.replace('/mnt/SeagateBackup_USB/Mutant_Movies/', '')
		videoname = videoname.replace('/mnt/SeagateBackup_USB/', '')
		videoname = videoname.replace('/', '--')
		videoname = videoname.replace('\r', '')
		return videoname

	def get_img_paths(self, img_folder):
		# Gather all .tif files (non-hidden) in the specified folder
		img_paths = []
		img_filenames = []
		for file in os.listdir(img_folder):
			if file.endswith('.tif') and not file.startswith('.'):
				img_paths.append(img_folder + '/' + file)
				img_filenames.append(file)
		# Sort images numerically if the #.tif file convention is used
		if bool(re.match('\d+.tif', img_filenames[0])):
			try:
				img_nums = [int(img_filename.replace('.tif', '')) for img_filename in img_filenames]
				# Then sort them numerically (not alphabetically, which is default)
				img_paths = np.array(img_paths)[np.argsort(img_nums)]
			except ValueError:
				print('Filenames inconsistent')
		# Sort images numerically if the #..tif file convention is used
		elif bool(re.match('\d+..tif', img_filenames[0])):
			try:
				img_nums = [int(img_filename.replace('..tif', '')) for img_filename in img_filenames]
				img_paths = np.array(img_paths)[np.argsort(img_nums)]
			except ValueError:
				print('Filenames inconsistent')
		# Sort images numerically if the Image#.tif file convention is used
		elif bool(re.match('Image\d+.tif', img_filenames[0])):
			try:
				img_nums = [int(img_filename[5:].replace('.tif', '')) for img_filename in img_filenames]
				img_paths = np.array(img_paths)[np.argsort(img_nums)]
			except ValueError:
				print('Filenames inconsistent')
		# Sort images numerically if the images#.tif file convention is used
		elif bool(re.match('images\d+.tif', img_filenames[0])):
			try:
				img_nums = [int(img_filename[6:].replace('.tif', '')) for img_filename in img_filenames]
				img_paths = np.array(img_paths)[np.argsort(img_nums)]
			except ValueError:
				print('Filenames inconsistent')
		# Sort images numerically if the EC_WT_##_####.tif file convention is used
		elif bool(re.match('EC_WT_\d\d_\d+.tif', img_filenames[0])):
			try:
				img_nums = [int(img_filename[-8:].replace('.tif', '')) for img_filename in img_filenames]
				img_paths = np.array(img_paths)[np.argsort(img_nums)]
			except ValueError:
				print('Filenames inconsistent')

		# Sort images numerically if the EC_WT_##_####.tif file convention is used
		elif bool(re.match('EC_WT_01_0 \(\d+\).tif', img_filenames[0])):
			try:
				img_nums = [int(re.findall('\((\d+)\)', img_filename)[0]) for img_filename in img_filenames]
				img_paths = np.array(img_paths)[np.argsort(img_nums)]
			except ValueError:
				print('Filenames inconsistent')

		# Accept the 3D scopes file convention
		elif bool(re.match('Run\d+_scope\d+-\d+_\d+', img_filenames[0])):
			pass
		# Give a warning if an unknown file convention was used
		else:
			print('Filename like "{}" not recognized'.format(img_filenames[0]))
		# Take only every tenth image
		img_paths = img_paths[::10]
		return img_paths

	def get_coordinates(self, contour):
		# Return the list of x,y coordinates for a given cv2 contour
		x = [pt[0][0] for pt in contour]
		y = [pt[0][1] for pt in contour]
		return(x,y)

	def warn_mkdir(self, dir):
		try:
			os.mkdir(dir)
		except FileExistsError:
			print('Directory ' + dir + ' already exists. Files may be overwritten')

	def getIDs(self, minarea=400, maxgray=200):
		# Load video data frames and combine them into one
		vid_df_paths = os.listdir(self.vid_df_dir)
		full_vid_df = pd.DataFrame(columns=['frame', 'x', 'y', 'area', 'perim', 'grayValue', 'circularity'])
		for vid_df_path in os.listdir(self.vid_df_dir):
			if vid_df_path.endswith('.txt'):
				vid_df = pd.read_csv(self.vid_df_dir + vid_df_path, sep='\t', index_col=0)
				full_vid_df = pd.concat((full_vid_df, vid_df))
		# NOTE: index column is contour number (for that frame)

		# Filter contours by area and gray value
		filtered_df = full_vid_df[(full_vid_df.area.values>minarea) & (full_vid_df.grayValue.values<maxgray)]
		all_contour_indices = filtered_df.index

		# Calculate and save the fruiting body IDs
		print('\tFull video data frame: {} rows'.format(len(filtered_df)))
		tracked = tp.link_df(filtered_df, 15, memory=7)
		tracked.to_csv(self.full_vid_df_path, sep='\t', float_format='%.3f')

		# Add category column to video df
		categorized = self.add_category_col(tracked)
		categorized.to_csv(self.cat_df_path, sep='\t', float_format='%.3f')

		return all_contour_indices,categorized

	def add_category_col(self, this_df):
		# Remove early identified FBs (spurious)
		early_p = np.unique(this_df[this_df.frame < 10].particle.values)
		nonearly_df = this_df[[p not in early_p for p in this_df.particle]]
		areas = nonearly_df.groupby('frame')['area'].agg(sum)

		# Identify persistent FBs
		persist_p = []
		for p,df_slice in nonearly_df.groupby('particle'):
			late_p = nonearly_df[nonearly_df.frame == max(nonearly_df.frame)].particle.values
			avgCirc = np.mean(df_slice.circularity)
			if (p in late_p) and (avgCirc > 0.5):
				persist_p.append(p)

		# Identify evaporators
		evap_p=[]
		for p,df_slice in nonearly_df.groupby('particle'):
			a = df_slice.area.values
			f = df_slice.frame.values
			xmin = min(df_slice.x.values)
			xmax = max(df_slice.x.values)
			ymin = min(df_slice.y.values)
			ymax = max(df_slice.y.values)
			notNearEdges = (xmin > 10) and (xmax < 1590) and (ymin > 10) and (ymax < 1190)
			avgCirc = np.mean(df_slice.circularity)
			if (a[-1] < 0.75*max(a)) and (min(f)>10) and notNearEdges and (avgCirc > 0.5) and (a[0] < max(a)) and (p not in persist_p):
				evap_p.append(p)

		# Add category column
		category_col = ['other']*len(this_df)
		for i,this_row in enumerate(this_df.iloc):
			if this_row.particle in persist_p:
				category_col[i] = 'persistent'
			elif this_row.particle in evap_p:
				category_col[i] = 'evaporates'
		this_df['category'] = category_col

		return this_df

	def read_batch(self, batch_file):
		with open(batch_file, 'r') as txtfile:
			img_folders = [txt.replace('\n', '') for txt in txtfile.readlines()]
		return img_folders

	##########################
	##########################
	###                    ### 
	### ANALYSIS FUNCTIONS ###
	###                    ###
	##########################
	##########################

	def class_default(func):
		# If the user doesn't specify a df, use the df
		# that is loaded into this class instance
		def wrapper(self, df=None, **kwargs):
			if df is None:
				df = self.df
			return func(self, df, **kwargs)
		return wrapper

	@class_default
	def get_start_time(self, df, n_threshold=10, area_threshold=970):
		# Return frame number when the number of fruiting bodies
		# above a certain size reaches a certain threshold
		def n_are_large(areas, threshold):
			return sum(np.array(areas) > threshold)
		df_slice = df[df.category != 'other']
		n_large = df_slice.groupby('frame')['area'].agg(n_are_large, area_threshold)
		times_when_n_large_exceeds = n_large.index[n_large >= n_threshold]
		if len(times_when_n_large_exceeds) == 0:
			return 0
		return min(times_when_n_large_exceeds)

	@class_default
	def get_peak_time(self, df):
		# Return frame number when max is hit for total area
		# counting only categorized fruiting bodies
		df_slice = df[df.category != 'other']
		areas = df_slice.groupby('frame')['area'].agg(sum)
		if len(areas) == 0:
			return 0
		times_at_max = areas.index[areas == max(areas)]
		return min(times_at_max)

	@class_default
	def get_growth_phase(self, df):
		# Return frame number difference between start and peak time
		return self.get_peak_time(df) - self.get_start_time(df)

	@class_default
	def get_avg_growth_rate(self, df):
		# Return average rate of total area increase between start and peak time
		df_slice = df[df.category != 'other']
		areas = df_slice.groupby('frame')['area'].agg(sum)
		if len(areas) == 0:
			return 0
		start_time = self.get_start_time(df)
		if start_time == 0: # Start time criterion never hit
			return 0
		d_area = max(areas) - areas.loc[start_time]
		d_time = self.get_peak_time(df) - start_time
		if d_time <= 0:
			return 0
		return d_area / d_time

	@class_default
	def get_area_std_at_peak_time(self, df):
		# Return standard deviation of fruiting body areas at peak time
		df_slice = df[df.category != 'other']
		peak_time = self.get_peak_time(df)
		return np.std(df_slice[df_slice.frame == peak_time].area.values)

	@class_default
	def get_final_area_std(self, df):
		# Return standard deviation of fruiting body areas in final frame
		df_slice = df[df.category != 'other']
		if len(df_slice) == 0:
			return 0
		return np.std(df_slice[df_slice.frame == max(df_slice.frame.values)].area.values)

	@class_default
	# NOTE: This metric assumes time HAS NOT been scaled to minutes!
	def get_gv_change(self, df, early_cutoff=50):
		# Return the percent change of average persistor gray value
		df_slice = df[df.category == 'persistent']
		if len(df_slice) == 0:
			return 0
		gvc = df_slice.groupby('frame')['grayValue'].agg(np.mean)
		gvc_slice = gvc[gvc.index > early_cutoff]
		if len(gvc_slice) == 0:
			return 0
		if max(gvc_slice) == 0:
			return 0
		return (max(gvc_slice) - min(gvc_slice))/max(gvc_slice)

	@class_default
	def get_gv_spatial_std(self, df):
		# Return
		pass

	@class_default
	def get_std_t_evap_max_area(self, df):
		df_slice = df[df.category == 'evaporates']
		if len(df_slice) == 0:
			return 0
		ts_max_area = []
		for p,fb_data in df_slice.groupby('particle'):
			t_max_area = min(fb_data[fb_data.area == max(fb_data.area)].frame)
			ts_max_area.append(t_max_area)
		return np.std(ts_max_area)	

	@class_default
	def get_frac_evap(self, df):
		# Return the fraction of all identified fruiting bodies
		# that evaporate
		n_evap = self.get_n_evap(df)
		n_pers = self.get_n_pers(df)
		if n_evap + n_pers == 0:
			return 0
		return n_evap/(n_evap + n_pers)

	@class_default
	def get_max_n(self, df):
		# Return total number of fruiting bodies present at the moment
		# of peak total area
		df_slice = df[df.category != 'other']
		if len(df_slice) == 0:
			return 0
		ns = df_slice.groupby('frame')['area'].agg(len)
		return max(ns.values)

	@class_default
	def get_avg_area_at_peak_time(self, df):
		# Return average fruiting body area at the moment of peak total area
		df_slice = df[df.category != 'other']
		if len(df_slice) == 0:
			return 0
		peak_time = self.get_peak_time(df)
		return np.mean(df_slice[df_slice.frame == peak_time].area.values)

	@class_default
	def get_final_avg_area(self, df):
		# Return average fruiting body area at 24 hour mark
		df_slice = df[df.category != 'other']
		if len(df_slice) == 0:
			return 0
		return np.mean(df_slice[df_slice.frame == max(df_slice.frame.values)].area.values)

	@class_default
	def get_final_std_area(self, df):
		# Return standard deviation of fruiting body area at 24 hour mark
		df_slice = df[df.category != 'other']
		if len(df_slice) == 0:
			return 0
		return np.std(df_slice[df_slice.frame == max(df_slice.frame.values)].area.values)

	@class_default
	# NOTE: This metric assumes time HAS NOT been scaled to minutes!
	def get_maturation_rate(self, df, early_cutoff=50):
		# Return maximum rate of gray value change in persistors,
		# normalized as a percent change (and sign flipped so positive values imply normal maturation)
		df_slice = df[df.category == 'persistent']
		gvc = df_slice.groupby('frame')['grayValue'].agg(np.mean)
		if len(gvc[gvc.index > early_cutoff]) == 0: # No persistors develop beyond the early cutoff
			return 0
		if len(gvc) < 41: # No persistors develop for more than 410 minutes
			return 0
		if max(gvc.values) == 0:
			return 0
		smoothed_deriv = savgol_filter(gvc.values, 41, 3, deriv=1)/max(gvc.values[gvc.index > early_cutoff])
		m_r = min(smoothed_deriv[gvc.index > early_cutoff])
		if m_r == smoothed_deriv[gvc.index > early_cutoff][0]: # If the first point was chosen, ignore the initial monotonic section of the curve
			early_cutoff_i = 0
			for i in range(1,len(gvc)):
				if smoothed_deriv[i] > smoothed_deriv[early_cutoff_i]:
					early_cutoff_i = i
				else:
					break
			early_cutoff = gvc.index[early_cutoff_i]
			m_r = min(smoothed_deriv[gvc.index > early_cutoff])
		return -m_r*100

	@class_default
	def get_mean_lifetime(self, df):
		lifetimes = []
		df_evap = df[df.category == 'evaporates']
		if df_evap.empty:
			return 0 
		for p,df_slice in df_evap.groupby('particle'):
			frame_i = min(df_slice.frame.values)
			frame_f = max(df_slice.frame.values)
			lifetimes.append(frame_f - frame_i)
		return np.exp(np.mean(np.log(lifetimes))) # Log of lifetime appears normally distributed
	
	@class_default
	def get_std_lifetime(self, df):
		lifetimes = []
		df_evap = df[df.category == 'evaporates']
		if df_evap.empty:
			return 0 
		for p,df_slice in df_evap.groupby('particle'):
			frame_i = min(df_slice.frame.values)
			frame_f = max(df_slice.frame.values)
			lifetimes.append(frame_f - frame_i)
		return np.exp(np.std(np.log(lifetimes))) # Log of lifetime appears normally distributed

	@class_default
	def get_stability_time(self, df, window=31, threshold=0.5):
		df_slice = df[df.category != 'other']
		n = df_slice.groupby('frame')['area'].agg(len)
		if len(n) < window:
			return 200
		n_deriv = savgol_filter(n.values, window, 3, deriv=1)
		if np.abs(n_deriv[-1]) < threshold: # stability is achieved
			for j in range(1,len(n_deriv)):
				if np.abs(n_deriv[-j]) >= threshold:
					break
			return n.index.values[-j]
		else:
			return 200 # A hypothetical eventual stability time?

	@class_default
	def get_max_avg_area_falloff(self, df, window=31):
		df_slice = df[df.category != 'other']
		avg_areas = df_slice.groupby('frame')['area'].mean()
		if len(avg_areas) < window:
			return 0
		a_deriv = savgol_filter(avg_areas.values, window, 3, deriv=1)
		return min(a_deriv)

	@class_default
	def get_max_n_falloff(self, df, window=31):
		df_slice = df[df.category != 'other']
		n = df_slice.groupby('frame')['area'].agg(len)
		if len(n) < window:
			return 0
		n_deriv = savgol_filter(n.values, window, 3, deriv=1)
		return min(n_deriv)

	@class_default
	def get_n_evap(self, df):
		df_slice = df[df.category == 'evaporates']
		return len(np.unique(df_slice.particle))
	
	@class_default
	def get_a_evap(self, df):
		df_slice = df[df.category == 'evaporates']
		areas = df_slice.groupby('frame')['area'].agg(sum)
		if len(areas) == 0:
			return 0
		else:
			return max(areas)
	
	@class_default	
	def get_avg_a_evap(self, df):
		df_slice = df[df.category == 'evaporates']
		areas = df_slice.groupby('particle')['area'].agg(max)
		if len(areas) == 0:
			return 0
		else:
			return np.mean(areas)
	
	@class_default
	def get_n_pers(self, df):
		df_slice = df[df.category == 'persistent']
		return len(np.unique(df_slice.particle))
	
	@class_default
	def get_a_pers(self, df):
		df_slice = df[df.category == 'persistent']
		areas = df_slice.groupby('frame')['area'].agg(sum)
		if len(areas) == 0:
			return 0
		else:
			return max(areas)
	
	@class_default	
	def get_avg_a_pers(self, df):
		df_slice = df[df.category == 'persistent']
		areas = df_slice.groupby('particle')['area'].agg(max)
		if len(areas) == 0:
			return 0
		else:
			return np.mean(areas)

	@class_default
	def total_area_curves(self, df):
		evap_df = df[df.category == 'evaporates']
		pers_df = df[df.category == 'persistent']
		other_df = df[df.category == 'other']
	
		# Total area of fruiting bodies over time
		evap_areas = evap_df.groupby('frame')['area'].sum()
		pers_areas = pers_df.groupby('frame')['area'].sum()
		other_areas = other_df.groupby('frame')['area'].sum()
		total_areas = pers_areas.add(evap_areas, fill_value=0)

		return evap_areas, pers_areas, total_areas, other_areas

	@class_default
	def total_number_curves(self, df):
		evap_df = df[df.category == 'evaporates']
		pers_df = df[df.category == 'persistent']
		other_df = df[df.category == 'other']
	
		# Total number of fruiting bodies over time
		evap_ns = evap_df.groupby('frame')['area'].agg(len)
		pers_ns = pers_df.groupby('frame')['area'].agg(len)
		other_ns = other_df.groupby('frame')['area'].agg(len)
		total_ns = pers_ns.add(evap_ns, fill_value=0)

		return evap_ns, pers_ns, total_ns, other_ns

	@class_default
	def fb_area_curves(self, df):
		evap_df = df[df.category == 'evaporates']

		curves = []
		for p,df_slice in evap_df.groupby('particle'):
			times = df_slice.frame.values
			areas = df_slice.area.values
			curve.append(p, times, areas)


	@class_default
	def avg_gray_curves(self, df, early_cutoff=50):
		evap_df = df[df.category == 'evaporates']
		pers_df = df[df.category == 'persistent']
		other_df = df[df.category == 'other']

		# Gray value of fruiting bodies over time
		evap_gray = evap_df.groupby('frame')['grayValue'].agg(np.mean)
		evap_gray = evap_gray[evap_gray.index > early_cutoff]

		pers_gray = pers_df.groupby('frame')['grayValue'].agg(np.mean)
		pers_gray = pers_gray[pers_gray.index > early_cutoff]

		other_gray = other_df.groupby('frame')['grayValue'].agg(np.mean)
		other_gray = other_gray[other_gray.index > early_cutoff]

		total_gray = pers_gray.add(evap_gray, fill_value=0)
		
		return evap_gray, pers_gray, total_gray, other_gray

	################
	################
	###          ###
	### PLOTTING ###
	###          ###
	################
	################

	def overlay_img(self, framenum, lw=5, include_other=False):
		img = cv2.imread(self.img_paths[framenum])
		# Reload points_path so you don't have to specify framenum in the fb_tools instance
		points_path = self.points_dir + self.videoname + '_{:04d}.npy'.format(framenum)
		points = np.load(points_path, allow_pickle=True)
		df = self.df[self.df.frame == framenum]
		if 'Unnamed: 0' in self.df.columns:
			evap_contours_i = df[df.category == 'evaporates']['Unnamed: 0'].values
			pers_contours_i = df[df.category == 'persistent']['Unnamed: 0'].values
			othr_contours_i = df[df.category == 'other']['Unnamed: 0'].values
		else:
			evap_contours_i = df[df.category == 'evaporates'].index.values
			pers_contours_i = df[df.category == 'persistent'].index.values
			othr_contours_i = df[df.category == 'other'].index.values
		evap_points = points[evap_contours_i]
		pers_points = points[pers_contours_i]
		othr_points = points[othr_contours_i]
		output = cv2.drawContours(img, evap_points, -1, (255,127,14), lw)
		output = cv2.drawContours(output, pers_points, -1, (44,160,44), lw)
		if include_other:
			output = cv2.drawContours(output, othr_points, -1, (255,255,14), lw)
		return output

	@class_default
	def metric_plots(self, df, fig=None, ax=None):
		if fig is None or ax is None:
			fig, ax = plt.subplots(4,1)
		
		evap_areas, pers_areas, total_areas, other_areas = self.total_area_curves(df)
		evap_ns, pers_ns, total_ns, other_ns = self.total_number_curves(df)
		evap_gray, pers_gray, total_gray, other_gray = self.avg_gray_curves(df)

		ax[0].plot(total_areas)
		ax[0].set_ylabel('Total area')

		ax[1].plot(total_ns)
		ax[1].set_ylabel('Total number')

		ax[2].plot(pers_gray, color='C2')
		ax[2].set_ylabel('Average gray value')