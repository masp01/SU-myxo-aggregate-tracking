import pandas as pd
import numpy as np
import cv2 #all image processing
import sys
from fb_tools import fb_tools

def processFrame(img_folder, framenum, denoising_strength=70, kernel_size=5, threshold_offset=20):
    # Prepare paths and info for this frame
    fb = fb_tools(img_folder, framenum)

    # Open image
    #
    print('Opening image')
    #
    frame = cv2.imread(fb.img_path, 0) # 0 opens image in grayscale

    # Nonlinear means denoising (blur that maintains edges)
    #
    print('Nonlinear means denoising')
    #
    dst = cv2.fastNlMeansDenoising(frame, None, denoising_strength, 7, 21)
    
    # Adaptive thresholding (makes binary image)
    #
    print('Thresholding')
    #
    thresh1=cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,threshold_offset) 
    
    # Morphological opening (denoises binary image)
    #
    print('Morphological opening')
    #
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    opened=cv2.morphologyEx(255-thresh1,cv2.MORPH_OPEN,kernel)
    
    # Find contours and save them to points_path (in .npy format)
    #
    print('Finding contours')
    #
    contours,h = cv2.findContours(opened,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    np.save(fb.points_path, np.array(contours, dtype=object))

    #
    print('Finished saving contours')

    def getAvgGray(frame, contour, contour_i):
        mask = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(mask,contours,contour_i,255,-1)
        return cv2.mean(frame,mask)[0]

    def getCenterOfMassCoords(contour, axis):
        return np.mean([contour[i][0][axis] for i in range(len(contour))])

    # Populate dataframe columns (vectorized) -- Wow, this is still by far the slowest step! Especially slow for very large images
    #
    print('Populating dataframe columns')
    #
    all_framenum = [framenum]*len(contours)
    all_cxs = [getCenterOfMassCoords(contour, 0) for contour in contours]
    all_cys = [getCenterOfMassCoords(contour, 1) for contour in contours]
    all_areas = [cv2.contourArea(contour) for contour in contours]
    all_perims = [cv2.arcLength(contour, True) for contour in contours]
    all_avggray = [getAvgGray(frame, contours[contour_i], contour_i) for contour_i in range(len(contours))]

    # Assemble and save dataframe
    #
    print('Saving dataframe')
    #
    vid_df = pd.DataFrame({
        'frame':all_framenum,
        'x':all_cxs,
        'y':all_cys,
        'area':all_areas,
        'perim':all_perims,
        'grayValue':all_avggray
        })
    # Calculate circularity column
    vid_df['circularity'] = 4*np.pi*vid_df.area.values / (vid_df.perim.values)**2
    vid_df.to_csv(fb.vid_df_path, sep='\t')
    

    # Create diagnostic image with contours drawn on
    #
    #img = cv2.drawContours(frame, contours, -1, 255, 2)
    #aspect = np.shape(frame)[0] / np.shape(frame)[1]
    #plt.subplots(figsize=(10,10*aspect))
    #plt.imshow(frame, cmap='gray')



    # Populate dataframe columns
    #for contour_i in range(len(contours)):
    #    print('Populating dataframes for contour {0} of {1}'.format(contour_i, len(contours)))
    #    contour=contours[contour_i]
    
        # Center of mass coordinates
    #    x = []
    #    y = []
    #    for coordinates in range(len(contour)):
    #        x.append(contour[coordinates][0][0])
    #        y.append(contour[coordinates][0][1])
    #        # These populate the columns of the perimeter point coordinates data frame
    #        all_contournum.append(contour_i)
    #        all_ys.append(contour[coordinates][0][1])
    #        all_xs.append(contour[coordinates][0][0])
    #    cx,cy = np.mean(x), np.mean(y)
        # Area
    #    contourarea=cv2.contourArea(contour)
        # Perimeter
    #    contourperim=cv2.arcLength(contour, True)
        # Average gray value
    #    mask=np.zeros(frame.shape, np.uint8) #np.uint8 is required to use cv2 functions
    #    cv2.drawContours(mask,contours,contour_i,255,-1) #-1 means fill in the contour instead of just drawing the border
    #    avggray=cv2.mean(frame,mask)[0] #cv2.mean returns four values, for some reason (all the rest are 0's)
        
        # Add data to columns
    #    all_framenum.append(framenum)
    #    all_areas.append(contourarea)
    #    all_perims.append(contourperim)
    #    all_avggray.append(avggray)
    #    all_cxs.append(cx)
    #    all_cys.append(cy)

if __name__ == "__main__":
    print('Starting frame ' + sys.argv[2])
    processFrame(sys.argv[1], int(sys.argv[2]))
    print('\tFrame ' + sys.argv[2] + ' complete.')