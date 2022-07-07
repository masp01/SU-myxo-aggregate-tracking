import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as qt
import PyQt5.QtCore as qtc
import os
import pandas as pd
from MyxoNavigator import MyxoViewer

# run from Anaconda Prompt as
#	ipython --pylab --gui=qt5
#	run Registry

class App:
	def __init__(self):
		# Load video registry
		self.registry = pd.read_csv('Registry.txt', sep='\t')

		self.listItems = []

		self.viewer = None

		if os.path.exists('/home/user/myxo-tracking/Hotspots.csv'):
			self.hotspots = pd.read_csv('/home/user/myxo-tracking/Hotspots.csv', index_col=0)

		self.startGUI() # Starting the GUI first allows the images to also be open simultaneously
		#self.showImages()

	def rowName(self, i):
		img_folder = self.registry.loc[i].img_folder
		mutant = self.registry.loc[i].mutant
		phenotype = self.registry.loc[i].phenotype
		descriptor = os.path.basename(img_folder)
		if 'MXAN_' not in descriptor:
			descriptor = '{}-{}'.format(self.registry.loc[i].strain, descriptor)
		if i in self.hotspots.index:
			mutant += '_hot'
		return '{:04d}:{} {:_^17} {}'.format(i, mutant, phenotype, descriptor)

	def getSelectedRows(self):
		indices = [i.row() for i in self.listWidget.selectedIndexes()]
		return indices

	def startSearch(self):
		self.listWidget.clearSelection()
		if len(self.entryWidget.text()) < 2:
			self.updateSelectedLabel()
			return
		items = self.listWidget.findItems('.*{}.*'.format(self.entryWidget.text()), qtc.Qt.MatchRegExp)
		if len(items) == 0:
			self.updateSelectedLabel()
			return
		for item in items:
			item.setSelected(True)
		self.listWidget.scrollToItem(items[0], qt.QAbstractItemView.PositionAtTop)
		self.updateSelectedLabel()

	def updateSelectedLabel(self, current=None, previous=None): # args only passed by self.listWidget.currentItemChanged
		qt.qApp.processEvents() # ensures the current GUI state is used
		n_sel = len(self.listWidget.selectedIndexes())
		self.selectedLabel.setText('{} selected'.format(n_sel))

	def clearSearch(self):
		# Blocking signals prevents the clear selection that would happen
		# if self.startSearch were triggered by a user
		self.entryWidget.blockSignals(True)
		self.entryWidget.setText('')
		self.entryWidget.blockSignals(False)

	def viewVideo(self):
		indices = self.getSelectedRows()
		if len(indices) == 0:
			return
		self.viewer = MyxoViewer(indices[0])

	def startGUI(self):
		# Make GUI
		self.root = qt.QApplication([]) # Can take startup parameters
		self.window = qt.QWidget()
		self.window.setWindowTitle('Video registry')
		self.window.setGeometry(15, 15, 400, 500)
		self.winLayout = qt.QGridLayout()
		self.window.setLayout(self.winLayout)

		# Create widgets
		self.displayButton = qt.QPushButton('View Video')
		self.displayButton.adjustSize()

		self.entryWidget = qt.QLineEdit()

		self.selectedLabel = qt.QLabel('0 selected')

		self.listWidget = qt.QListWidget()
		self.listWidget.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
		for i in self.registry.index:
			self.listItems.append(qt.QListWidgetItem(self.rowName(i)))
			self.listWidget.addItem(self.listItems[i])
		
		# Position widgets
		self.winLayout.addWidget(self.displayButton, 0, 0)
		self.winLayout.addWidget(self.entryWidget, 1, 0)
		self.winLayout.addWidget(self.selectedLabel, 2, 0)
		self.winLayout.addWidget(self.listWidget, 3, 0)
		#self.button_one = qt.QPushButton('New Bucket')
		#self.button_one.clicked.connect(self.newBucket)
		#self.winLayout.addWidget(self.button_one, 0, 0)

		# Connect signals
		self.displayButton.clicked.connect(self.viewVideo)
		self.entryWidget.textChanged.connect(self.startSearch)
		self.listWidget.currentItemChanged.connect(self.updateSelectedLabel, type=qtc.Qt.QueuedConnection)
		self.listWidget.itemClicked.connect(self.clearSearch) # Clear search when manual selection changes are made

		# Start GUI
		self.window.show()
		#self.root.exec_() # Application loop, use when running the script stand-alone

if __name__=="__main__":
	app = App()