## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[cam_undistorted]: ./output_images/example/undistort_output.jpg "Undistorted"
[test_undistorted]: ./output_images/example/undistort.jpg "Undistorted"
[combined_binary]: ./output_images/example/combined_binary.jpg "Combined Binary"
[binary_warped]: ./output_images/example/binary_warped.jpg "Binary Warped"
[step1]: ./output_images/example/step1.jpg "Binary Warped"
[step2]: ./output_images/example/step2.jpg "Poly Fit"
[step3]: ./output_images/example/step3.jpg "Result"

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "P2.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][cam_undistorted]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
![alt text][test_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 4 through 5 ).
After experimenting with a combination of gabs_sobel_thresh (x and y), mag_thresh, dir_threshold and hls_threshold, I settled on a combination of 
sobel_x,mag_thresh and a new method i call alternative sobel_x( where i used the s channel in hls instead of gray image for the kernel convolution)

    combined = np.zeros_like(gradx)
    combined[ ((mag_binary == 1) & (gradx == 1))] = 1
    combined[ ((combined == 1) | (gradxAlt == 1))] = 1

 Here's an example of my output for this step.
![alt text][combined_binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_transform()`, which appears in cell 6 of the IPython notebook.  The `warper()` function takes as inputs an image (`img`) and an (`offset`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 57, img_size[1] / 2 + 97],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 97]])
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 583, 457      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1126, 720     | 960, 720      |
| 700, 457      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][binary_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The code for finding lane pixels uses a function called `find_lane_pixels` found in cell 9 of the IPython notebook. `find_lane_pixels` takes in a (`binary_warped`) image 
uses a histogram of the sum of pixels on the vertical axis to detect the lanes under the hood in the image. Then with a sliding widow, we trace the groups of pixels along the 
vertical axis to detect both left and right lane lines. 
Then we fit the detected lane pixels with a 2nd order polynomial kinda like this:

```python
 # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```
![alt step3][step1]
![alt step2][step2]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cells 10 line 32 of the IPython notebook. The code for calculating the curvature uses a function called `measure_curvature_real` 
which accepts the `x` and `y` pixel values and converts from pixels space to meters. The image center and offset from lane center is calculated
using the code below.

```python
# Calculate the radius of curvature in meters for both lane lines
    left_curverad, right_curverad = measure_curvature_real(left_fitx,right_fitx,ploty)
    avg_curve = (left_curverad+right_curverad)/2

    # calculate offset from center
    # lane_center = (right_fitx[0]+left_fitx[0])/2
    left_value = left_fit[0]*image.shape[0]**2 + left_fit[1]*image.shape[0] + left_fit[2]
    right_value = right_fit[0]*image.shape[0]**2 + right_fit[1]*image.shape[0] + right_fit[2]
    lane_center = (right_value+left_value)/2
    xm_per_pix = 3.7/700
    image_center = image.shape[1]/2
    offset_center = round((image_center-lane_center)*xm_per_pix,3)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 7 line 82 in the function `map_lane()`.  Here is an example of my result on a test image:

![alt step3][step3]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why.
The major factor i found in detecting the lanes correctly is getting a good binary image that emphasises the lane pixels
as much as possible. Another problem was whenever a car drove by in the next lane, close to my lane lines, the sliding window
includes the car pixels results in a wrong lane detection. To resolve this i had to reduce the size of the sliding window on the
function `search_around_poly` from 100 to 70 to narrow the boundary for the sliding window.
This pipeline will fail in generating reliable `binary_images` with emphasises on lane lines, other crossing 
vechiles and repaved road edges could be included in the edge detection phase. I think this could be improved by 
adding color masking mainly to detect the yellow lane lines, masking of white lines in addition to the combined_binary
could help improve highlighting of lane lines for detection.
