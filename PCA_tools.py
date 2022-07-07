import numpy as np
from fb_tools import fb_tools
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.patches import Patch
import cv2

registry = fb_tools().registry # for convenience when using `run PCA_tools`

# Calculate a new PCA from scratch:
#	run PCA_tools
#	sample = registry[...conditions...]
#	p = PCA_tools().new(sample) # Loads default metrics and scales them unsupervised
#	p.computePCA()

# Save data:
#	p.save() # Prompts user for savepath
#		or
#	p.save('/user/home/myxo-tracking/PCA/tmp/')

# Load previously calculated data:
#	data = PCA_tools().load() # Prompts user for loadpath
#		or
#	data = PCA_tools().load('/user/home/myxo-tracking/PCA/PCA_all/')
#
# Loading data also makes plotting functions available

# If computing new default metrics:
#	p = PCA_tools()
#	p.new(filter_errors=False)
#	p.computeMetrics()
#	
#	then place metrics_df.csv in the default parentPath

class PCA_tools:
	def __init__(self, parentPath='/home/user/myxo-tracking/PCA/'):
		# Filenames for all ouput
		# Keys are properties of the class
		self.output_filenames = {
			'registry_sample':'included_videos_unfiltered.csv',
			'filtered_registry_sample':'included_videos_filtered.csv',
			'metrics_df':'metrics_df.csv',
			'scaled_metrics_df':'scaled_metrics_df.csv',
			'principalDf':'PCA.csv',
			'pc_variance_ratios':'PC_variance.txt',
			'pc_metrics':'PC_metrics.txt',
			'nan_indices':'nan_indices.txt',
			'metrics_datetime':'when_metrics_calculated.txt'
			}

		# Initialize
		self.metric_names = np.array(['start_time', 'peak_time', 'stability_time', 'growth_time', 'growth_rate', 'peak_avg_area', 'peak_area_std', 'final_avg_area', 'final_area_std', 'gv_change', 'gv_rate', 'std_t_evap_max_area', 'frac_evap', 'max_n', 'mean_lifetime', 'std_lifetime', 'max_avg_area_falloff', 'max_n_falloff'])
		self.h_colors = {'ABC':'C0', 'Ntr-C':'C1', 'ECF':'C2', '1CN':'C6'} # Default homologous group colors

		self.parentPath = parentPath # Directory for output folders, should also contain metrics_df.csv for default pre-calculated metrics
		self.savepath = None

		self.registry_sample = None
		self.filtered_registry_sample = None
		self.metrics_df = None
		self.scaled_metrics_df = None
		self.principalDf = None
		self.nan_indices = None
		self.n_components = None
		self.metrics_datetime = None

		self.registry = fb_tools().registry

	# Prepares scaled metrics from registry sample
	# ready to run PCA
	def new(self, registry_sample=None, filter_errors=True, parentPath=None):
		# Default to using entire registry
		if registry_sample is None:
			registry_sample = self.registry

		# Clean up registry sample
		self.registry_sample, self.filtered_registry_sample = self.prepRegistrySample(registry_sample, filter_errors)

		# Load default metrics
		self.metrics_df = self.loadDefaultMetrics(parentPath)
		if self.metrics_df is None:
			# Check if some default metrics are missing
			return self

		# Scale metrics (unsupervised)
		self.scaleMetrics() # Populates self.scaled_metrics_df and self.nan_indices

		return self

	def save(self, savepath=None):
		# Prompt user for savepath if one is not given or previously specified
		if savepath is None:
			if self.savepath is None:
				savepath = self.makeNewDir(self.parentPath)
			else:
				savepath = self.savepath

		# Prompt user if anything might be overwritten
		proceed, savepath = self.checkSaveConflicts(savepath)
		if not proceed:
			return

		# Save each output file
		self.registry_sample.to_csv(savepath + self.output_filenames['registry_sample'])
		self.filtered_registry_sample.to_csv(savepath + self.output_filenames['filtered_registry_sample'])
		self.appendColumns(self.metrics_df).to_csv(savepath + self.output_filenames['metrics_df'])
		self.appendColumns(self.scaled_metrics_df).to_csv(savepath + self.output_filenames['scaled_metrics_df'])
		self.principalDf.to_csv(savepath + self.output_filenames['principalDf'])
		np.savetxt(savepath + self.output_filenames['pc_variance_ratios'], self.pc_variance_ratios)
		np.savetxt(savepath + self.output_filenames['pc_metrics'], self.pc_metrics)
		np.savetxt(savepath + self.output_filenames['nan_indices'], self.nan_indices, fmt='%d')
		with open(savepath + self.output_filenames['metrics_datetime'], 'w') as f:
			f.write(self.metrics_datetime)

		print('Output data saved to ' + savepath)
		self.savepath = savepath

	# Loads data saved at loadpath
	def load(self, loadpath=None):
		# Defaults to user prompt for loadpath
		if loadpath is None:
			loadpath = self.pickLoadpath(self.parentPath)

		# Check for each file in self.output_filenames
		for variable_name in self.output_filenames.keys():
			output_filename = self.output_filenames[variable_name]
			if os.path.exists(loadpath + output_filename):
				# Load in dataframes
				if output_filename.endswith('.csv'):
					df = pd.read_csv(loadpath + output_filename, index_col=0)
					if variable_name != 'principalDf':
						# Remove non-numeric columns that help human readability
						df = df.drop('mutant', axis=1)
						df = df.drop('strain', axis=1)
						df = df.drop('videoname', axis=1)
					# Store data in memory
					setattr(self, variable_name, df)
				# Load in metrics_datetime
				elif variable_name == 'metrics_datetime':
					with open(loadpath + output_filename, 'r') as f:
						self.metrics_datetime = f.readline()[:-1]
				# Load in numeric txt files
				else:
					if os.stat(loadpath).st_size != 0: # suppress an annoying warning about an empty file
						data = np.loadtxt(loadpath + output_filename)
					else:
						data = []
					# Store data in memory
					setattr(self, variable_name, data)
			else:
				print(loadpath + output_filename + ' missing')

		# Load the number of primary components
		self.n_components = sum(['pc' in colName for colName in self.principalDf.columns])
		return self

	#################
	#################
	###           ###
	###  PCA CORE ###
	###           ###
	#################
	#################

	def computeMetric(self, identifier):
		# Calculate all metric values for this video
		fb = fb_tools(identifier)
		metric_vector = []
		metric_vector.append(fb.get_start_time(fb.df))
		metric_vector.append(fb.get_peak_time(fb.df))
		metric_vector.append(fb.get_stability_time(fb.df))
		metric_vector.append(fb.get_growth_phase(fb.df))
		metric_vector.append(fb.get_avg_growth_rate(fb.df))
		metric_vector.append(fb.get_avg_area_at_peak_time(fb.df))
		metric_vector.append(fb.get_area_std_at_peak_time(fb.df))
		metric_vector.append(fb.get_final_avg_area(fb.df))
		metric_vector.append(fb.get_final_area_std(fb.df))
		metric_vector.append(fb.get_gv_change(fb.df))
		metric_vector.append(fb.get_maturation_rate(fb.df))
		metric_vector.append(fb.get_std_t_evap_max_area(fb.df))
		metric_vector.append(fb.get_frac_evap(fb.df))
		metric_vector.append(fb.get_max_n(fb.df))
		metric_vector.append(fb.get_mean_lifetime(fb.df))
		metric_vector.append(fb.get_std_lifetime(fb.df))
		metric_vector.append(fb.get_max_avg_area_falloff(fb.df))
		metric_vector.append(fb.get_max_n_falloff(fb.df))
		return metric_vector

	def computeMetrics(self, video_identifiers=None):
		# Default to indices in the filtered_registry_sample
		if video_identifiers is None:
			video_identifiers = self.filtered_registry_sample.index

		# Calculate and store metrics for each video
		metric_vectors = []
		for i in video_identifiers:
			# Indicate which metric is being calculated (in case of warnings/errors)
			videoname = fb_tools(i).videoname
			print('{}: {}'.format(i, videoname))

			# Calculate all metric values for this video
			metric_vector = self.computeMetric(i)

			# Store metric values for this video
			metric_vectors.append(metric_vector)

		# Keep date and time for when metrics were calculated
		self.metrics_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

		# Compile all metric values into a dataframe by video index
		self.metrics_df = pd.DataFrame(data=metric_vectors, columns=self.metric_names)
		self.metrics_df.index = video_identifiers

		return self.metrics_df

	# Filter out metrics that are nan
	# Scale remaining metrics unsupervised,
	#	i.e. mean -> 0, variance -> 1 for the given sample of metrics
	# Returns:
	# - dataframe of scaled metrics
	# - list of video indices with nan metrics
	def scaleMetrics(self, metrics_df=None):
		# Default to self.metrics_df
		if metrics_df is None:
			metrics_df = self.metrics_df

		# Remove vectors that have missing values
		metrics_df = self.filterNan(metrics_df, store_nan_indices=True)

		# Scale each metric to have mean 0 and variance 1
		scaled_metric_vectors = StandardScaler().fit_transform(metrics_df)

		# Compile all scaled metric values into a dataframe by video index
		self.scaled_metrics_df = pd.DataFrame(data=scaled_metric_vectors, columns=self.metric_names)
		self.scaled_metrics_df.index = metrics_df.index

		return self.scaled_metrics_df, self.nan_indices

	def computePCA(self, input_data=None, n_components=6):
		self.n_components = n_components

		# Default to using the scaled metric vectors as input
		if input_data is None:
			if self.scaled_metrics_df is None:
				self.scaleMetrics()
			input_data = self.scaled_metrics_df

		# Report on how many mutants are included in the PCA
		print(self.mutantCount(input_data))

		# Perform PCA
		pca = PCA(n_components=n_components)
		pc = pca.fit_transform(input_data)
		pca_out = pca.fit(input_data)		

		# Save output:
		# - principal component values for each video
		# - explained variance of each PC
		# - metric makeup of each PC
		self.principalDf = pd.DataFrame(data = pc, columns = ['pc{}'.format(i+1) for i in range(n_components)])
		self.principalDf.index = input_data.index
		self.principalDf = self.appendColumns(self.principalDf)

		self.pc_variance_ratios = pca_out.explained_variance_ratio_
		self.pc_metrics = pca_out.components_
		#self.covariance = pca_out.get_covariance()

		# Report on explained variance
		print('Explained variance:')
		print(self.pc_variance_ratios)
		print('{}% of total variance explained by the top {} primary components'.format(int(100*sum(self.pc_variance_ratios)), n_components))

	###########################
	###########################
	###                     ###
	###  PLOTTING FUNCTIONS ###
	###                     ###
	###########################
	###########################

	def PCAcoords(self, mutant, column='pc1', data=None):
		if data is None:
			data = self.principalDf
		return data[data.mutant == mutant][column].values

	# Scatterplot of all PCA points
	def plotPCA(self, x_axis='pc1', y_axis='pc2', data=None, fig=None, ax=None):
		# Make new plot window if one isn't supplied
		if (fig is None) or (ax is None):
			self.fig,self.ax = plt.subplots()
		else:
			self.fig = fig
			self.ax = ax

		# Use full principalDf if a slice isn't supplied
		if data is None:
			data = self.principalDf

		# Plot PCA output
		self.fig.canvas.manager.set_window_title("PCA Feature Space")
		self.ax.set_aspect('equal')
		for i in data.index:
			homolog_group = data.loc[i,'mutant']
			x,y = data.loc[i, x_axis], data.loc[i, y_axis]
			self.ax.scatter(x,y, color=self.h_colors[homolog_group])

		self.fig.tight_layout()
		plt.show()

	def densityMap(self, xs, ys, extent=(-6.5,6.5), nBins=20, kernelWidth=5):
		heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=np.linspace(*extent, nBins))
		density = cv2.GaussianBlur(heatmap, (kernelWidth, kernelWidth), 0)
		return density

	def plotDensity(self, mutant=None, scatter=True, scattersize=2, x_axis='pc1', y_axis='pc2', data=None, fadeExp=2, darkenFactor=0.75, scatter_alpha=0.6, extent=(-6.5, 6.5), nBins=20, kernelWidth=5, colorbar=True, fig=None, ax=None):
		# Make new plot window if one isn't supplied
		if (fig is None) or (ax is None):
			self.fig,self.ax = plt.subplots()
		else:
			self.fig = fig
			self.ax = ax

		# Use full principalDf if a slice isn't supplied
		if data is None:
			data = self.principalDf

		# Filter by mutant if supplied
		if mutant is not None:
			data = data[data.mutant == mutant]
			scatter_color = self.darken(self.h_colors[mutant], darkenFactor)
			fade_map = self.fade(self.h_colors[mutant], fadeExp)
		else:
			scatter_color = self.darken('C0', darkenFactor)
			fade_map = self.fade('C0', fadeExp)

		# Extract data
		xs,ys = data[x_axis], data[y_axis]

		# Make scatterplot, if desired
		if scatter:
			self.ax.scatter(xs, ys, scattersize, color=scatter_color, alpha=scatter_alpha)
		
		# Plot density
		self.ax.set_xlim(*extent)
		self.ax.set_ylim(*extent)
		im = self.ax.imshow(self.densityMap(xs, ys, extent=extent, nBins=nBins, kernelWidth=kernelWidth).T, extent=(*extent,*extent), origin='lower', cmap=fade_map)
		if colorbar:
			self.fig.colorbar(im)

	def plotPCmakeup(self, top_n_components=4, top_n_metrics=10, width_ratios=[1,5], fig=None, ax=None):
		# Make new plot window if one isn't supplied
		if (fig is None) or (ax is None):
			self.fig,self.ax = plt.subplots(1,2, figsize=(8,4), gridspec_kw={'width_ratios': width_ratios})
		else:
			self.fig = fig
			self.ax = ax

		# Calculate PCA if it has not been done yet
		if self.principalDf is None:
			self.computePCA()

		###########################
		# Explained variance plot #
		###########################
		# Colors for explained variance bars
		bar_colors = [plt.cm.Blues(i) for i in np.linspace(0.25, 1, self.n_components)]
		
		total_explained_variance = sum(self.pc_variance_ratios)
		# Bar for "other"
		self.ax[0].bar(0, 1-total_explained_variance, bottom=total_explained_variance, fill=None, edgecolor='lightgray', hatch='//')
		# Stacked bars the explained variance of each PC
		for i in range(self.n_components):
			self.ax[0].bar(0, self.pc_variance_ratios[i], bottom=sum(self.pc_variance_ratios[:i]), color=bar_colors[i], edgecolor='black')

		# Add text to label the variance bars
		i=0
		for bar in self.ax[0].patches:
			if i > 0:
				self.ax[0].text(
					# Center text
					bar.get_x() + bar.get_width() / 2,
					# Offset text vertically
					bar.get_height() + bar.get_y()-0.055,
					'PC{}'.format(i),
					ha='center',
					color='w',
					weight='bold',
					size=12
				)
			i += 1

		# Style axes
		self.ax[0].axes.xaxis.set_visible(False)
		self.ax[0].set_xlim(-1,1)
		self.ax[0].set_yticks(np.arange(0, 1.1, 0.1))
		self.ax[0].set_ylabel('Cumulative variance')
		self.ax[0].set_title('Variance explained')

		#########################################
		# Primary component breakdown by metric #
		#########################################
		# Choose 18 distinct colors, with the 10 darkest colors the top metrics of PC1,
		# and the other 8 pastel
		init_colors = np.concatenate((plt.cm.tab10.colors, plt.cm.Pastel2.colors))[:18]
		pc1_sort = np.argsort(np.abs(self.pc_metrics[0]))[::-1]
		metric_colors = np.array([None]*18)
		for i in range(18):
			metric_colors[pc1_sort[i]] = init_colors[i]
		
		for i in range(top_n_components):
			pci_sort = np.argsort(np.abs(self.pc_metrics[i]))[::-1] # Sort descending
			pci_sort_comp = self.pc_metrics[i][pci_sort]
			pci_sort_names = self.metric_names[pci_sort]
			bar_norm = sum(np.abs(pci_sort_comp))
			self.ax[1].bar('PC{}'.format(i+1), sum(np.abs(pci_sort_comp)[top_n_metrics:])/bar_norm, fill=None, edgecolor='lightgray', hatch='//')
			for j in range(top_n_metrics):
				bar,= self.ax[1].bar('PC{}'.format(i+1), np.abs(pci_sort_comp[j])/bar_norm, bottom=sum(np.abs(pci_sort_comp)[(j+1):])/bar_norm, color=metric_colors[pci_sort][j], edgecolor='black')
				if pci_sort_comp[j] < 0:
					self.ax[1].scatter(bar.get_x()+bar.get_width()/8, bar.get_y()+bar.get_height()/2, color='black', zorder=3)
		self.ax[1].set_ylim(0, 1.1)
		self.ax[1].axes.yaxis.set_visible(False)
		
		# Shrink current axis by 20%
		box = self.ax[1].get_position()
		self.ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
		
		# Put a legend to the right of the current axis
		legend_elements = [Patch(facecolor=metric_colors[pc1_sort][i], edgecolor='black', label=self.metric_names[pc1_sort][i]) for i in range(18)]
		self.ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=legend_elements)
		self.ax[1].set_title('Top metrics of Principal Components\n(A dot indicates the negative direction)')

	# Graphical helper functions
	def fade(self, color, exponent=1):
		if isinstance(color, str):
			color = colors.to_rgba(color)
		if len(color) == 3: # no alpha in specified color
			newColors = [(*color, a) for a in np.linspace(0,1,256)**exponent]
		elif len(color) == 4:
			r,g,b,a = color
			newColors = [(r,g,b,alpha) for alpha in np.linspace(0,1,256)**exponent]
		return colors.ListedColormap(newColors)

	def darken(self, color, factor=0.5):
		if isinstance(color, str):
			color = colors.to_rgba(color)
		if len(color) == 3: # no alpha in specified color
			r,g,b = color
			a = 1
		elif len(color) == 4:
			r,g,b,a = color
		return (r*factor, g*factor, b*factor, a)

	##########################
	##########################
	###                    ###
	###  HELPER FUNCTIONS  ###
	###                    ###
	##########################
	##########################

	def loadDefaultMetrics(self, parentPath=None):
		# Prepare directory that contains precalculated metric vectors
		if parentPath is None:
			parentPath = self.parentPath
		parentPath = self.ensureFinalSlash(parentPath)

		# Ensure metrics_df.csv is in the parentPath
		default_metrics_path = parentPath + self.output_filenames['metrics_df']
		if not os.path.exists(default_metrics_path):
			print('Precalculated metrics at ' + default_metrics_path + ' do not exist')
			return

		# Load default precalculated metric vectors
		self.metrics_df = pd.read_csv(default_metrics_path, index_col=0)
		self.metrics_df = self.metrics_df[self.metric_names]
		print('Loaded precalculated metrics')

		# Get datetime string for when these metrics were calculated
		timestamp = datetime.fromtimestamp(os.path.getmtime(default_metrics_path))
		self.metrics_datetime = timestamp.strftime("%Y-%m-%d, %H:%M:%S")
		
		# Restrict to videos for the given registry sample
		try:
			self.metrics_df = self.metrics_df.loc[self.filtered_registry_sample.index]
		except KeyError:
			# The registry sample contains indices for videos with unprocessed metrics
			print('\n***New metrics must be calculated***')
			print('\tp = PCA_tools()')
			print('\tp.new(filter_errors=False)')
			print('\tp.computeMetrics()')
			print('\n\tthen place metrics_df.csv in the default parentPath')
			return

		return self.metrics_df

	# Add mutant, strain, and videoname columns to a dataframe
	# Requires df.index to be registry index values
	def appendColumns(self, df):
		new_df = df.copy()
		mutants = [self.registry.loc[i,'mutant'] for i in new_df.index]
		strains = [self.registry.loc[i,'strain'] for i in new_df.index]
		videonames = [self.registry.loc[i,'videoname'] for i in new_df.index]
		if 'videoname' not in new_df.columns:
			new_df.insert(0, 'videoname', videonames)
		if 'strain' not in new_df.columns:
			new_df.insert(0, 'strain', strains)
		if 'mutant' not in new_df.columns:
			new_df.insert(0, 'mutant', mutants)
		return new_df

	# Prints a summary of processing errors
	def errorSummary(self, print_summary=True):
		if len(self.filtered_registry_sample) == len(self.registry_sample):
			errorString = fb_tools().errorSummary(self.filtered_registry_sample, print_summary=False)
		else:
			errorString = '***Unfiltered videos:***\n'
			errorString += fb_tools().errorSummary(self.registry_sample, print_summary=False)
			errorString += '\n\n***Filtered videos:***\n'
			errorString += fb_tools().errorSummary(self.filtered_registry_sample, print_summary=False)

		if self.nan_indices is not None:
			errorString += '\n\n{} videos removed due to missing metric values (indices in self.nan_indices)'.format(len(self.nan_indices))

		if print_summary:
			print(errorString)
		else:
			return errorString

	# Returns whether to proceed with saving and where
	# based on whether conflicts exist
	# and if so what the user specifies
	def checkSaveConflicts(self, savepath):
		# Prompt user if anything will be overwritten
		conflictingFiles = [f for f in self.output_filenames.values() if os.path.exists(savepath + f)]
		if len(conflictingFiles) == 0:
			proceed = True
		else:
			print('The following files will be overwritten in {}:'.format(savepath))
			for f in conflictingFiles:
				print('\t' + f)
			overwriteAnswer = '?'
			while overwriteAnswer not in ['y', 'n', '']:
				overwriteAnswer = input('Do you want to overwrite these files? y/[n] ')
			# Give user the option to save in a new directory
			if overwriteAnswer != 'y':
				newDirAnswer = '?'
				while newDirAnswer not in ['y', 'n', '']:
					newDirAnswer = input('Do you want to save in a new directory? [y]/n ')
				if newDirAnswer != 'n':
					savepath = self.makeNewDir(self.parentPath)
					proceed = True
				# Otherwise abort save
				else:
					proceed = False
			else:
				proceed = True

		return proceed, savepath

	def filterNan(self, df, store_nan_indices=False, return_nan_indices=False):
		# Remove vectors that have missing values
		filtered_df = df.dropna()
		kept_indices = filtered_df.index
		if len(df) != len(filtered_df):
			print('{} rows of {} removed due to missing values'.format(len(df)-len(filtered_df), len(df)))
		
		# Store indices of which vectors were nan, if desired
		if store_nan_indices:
			self.nan_indices = [i for i in df.index if i not in kept_indices]
		
		# Return indices of which vectors were nan, if desired
		# Otherwise, just return the filtered dataframe
		if return_nan_indices:
			return nan_indices, filtered_df
		else:
			return filtered_df

	def mutantCount(self, df=None):
		if df is None:
			df = self.filtered_registry_sample
		# Print how many videos of each type were included
		mutants = np.array([self.registry.loc[i,'mutant'] for i in df.index])
		mutant_counts = {h:(sum(mutants == h)) for h in self.h_colors.keys()}
		statusString = ''
		for h in self.h_colors.keys():
			statusString += '{} {} videos, '.format(mutant_counts[h], h)
		statusString += '\n{} videos total'.format(len(df))
		return statusString

	def makeNewDir(self, parentPath=None):
		# Prepare parent directory for new output folder
		if parentPath is None:
			parentPath = self.parentPath
		parentPath = self.ensureFinalSlash(parentPath)

		# Make folder for new output
		newDirSuccess = False
		while not newDirSuccess:
			# Prompt user for new output folder name
			newDirName = input('New output folder name? ')
			# Make the new folder if possible
			if not os.path.exists(parentPath + newDirName):
				try:
					os.mkdir(parentPath + newDirName)
					newDirSuccess = True
					print('Folder ' + parentPath + newDirName + ' created')
				except:
					print('Folder ' + parentPath + newDirName + ' could not be created')
			else:
				# Allow an empty, preexisting folder to be used
				if len(os.listdir(parentPath + newDirName)) == 0:
					newDirSuccess = True
				else:
					print('Folder ' + parentPath + newDirName + ' already exists and is not empty')

		return self.ensureFinalSlash(parentPath + newDirName)

	def pickLoadpath(self, parentPath=None):
		# Get a parentPath containing the available loadpath folders
		if parentPath is None:
			parentPath = self.parentPath
		parentPath = self.ensureFinalSlash(parentPath)

		# Get all folders in the parentPath
		allContents = os.listdir(parentPath)
		allFolders = [f for f in allContents if os.path.isdir(parentPath + f)]
		if len(allFolders) == 0:
			print('No folders in directory ' + parentPath)
			return

		# Prompt user to pick a folder
		print('Available folders:')
		for f in allFolders:
			print('\t'+f)
		folderPick = '?'
		while not os.path.exists(parentPath + folderPick):
			folderPick = input('')

		return self.ensureFinalSlash(parentPath + folderPick)

	def prepRegistrySample(self, registry_sample=None, filter_errors=True):
		# Default to using all videos if no registry sample is given
		if registry_sample is None:
			registry_sample = self.registry

		# Ignore videos that are unprocessed
		# If desired, also filter out videos with non-fatal errors
		if filter_errors:
			filtered_registry_sample = registry_sample[(registry_sample.processed == True) & (registry_sample.error == 0)]
			if len(filtered_registry_sample) != len(registry_sample):
				print('{} videos of {} with nonzero error codes ignored (summary in self.errorSummary)'.format(len(registry_sample)-len(filtered_registry_sample), len(registry_sample)))
		else:
			# An odd error code indicates a fatal error (missing data for the video)
			filtered_registry_sample = registry_sample[registry_sample.processed & (registry_sample.error%2 == 0)]
			if len(filtered_registry_sample) != len(registry_sample):
				print('{} videos of {} with fatal error codes ignored (summary in self.errorSummary)'.format(len(registry_sample)-len(filtered_registry_sample), len(registry_sample)))
		

		return registry_sample, filtered_registry_sample

	def ensureFinalSlash(self, path):
		if path[-1] != '/':
			path += '/'
		return path