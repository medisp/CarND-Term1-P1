import math





# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

def readimages(dir):
    yi = 0
    image=[]
    for i in os.listdir(dir):
        globals()['string%s' % y] = 'Hello'       
        i2=dir+i
        globals()['image%s' % y] = mpimg.imread(i2)
        yi+=1
    imagelist=[]
    for i in range(yi):
        imagelist.append('image%s' % i)
    return(imagelist)    

def processimages(imagelist):
    yp=0
    for i in imagelist:
        print(i)
        plt.imshow(eval(i))
        globals()['imagegray%s' % yp] = grayscale(eval(i))
        globals()['vertex%s' % count] = np.array( [[[125,eval(i).shape[0]],[450,325],[500,325],[875,eval(i).shape[0]]]], dtype=np.int32 )
		print(vertex1)
		
		yp+=1
 
    print('This image is:',i,type(image4), 'with dimensions:', image4.shape[0]) 
    vertices = np.array( [[[125,image4.shape[0]],[450,325],[500,325],[875,image4.shape[0]]]], dtype=np.int32 )
    kernelsize=3
    
    
    #gaussian4=gaussian_blur(canny(region_of_interest(image4,a3),50,250),kernelsize)
    gaussian4 = gaussian_blur(canny(image4,50,250),kernelsize)
    rho=2
    theta=40
    threshold=5
    min_line_len=3
    max_line_gap=5
    gaussian4 = gaussian_blur(canny(image4,50,250),kernelsize)
    
    #h4= hough_lines(gaussian4,rho=2,theta=10,threshold=4,min_line_len=4,max_line_gap=4) h4=hough_lines(gaussian4,2,1,12,3,25)
    
    h4=hough_lines(gaussian4,10,1,50,18,10)
    #plt.imshow(h4)
    
    roi2_4= hough_lines(canny(region_of_interest(h4,vertices),50,250),5,1,10,5,180)
    plt.imshow(roi2_4)
    plt.imshow(cv2.addWeighted(image4,0.8,roi2_4,1,0))
    
   
    print(image5.shape,imagegray5.shape)
    
    
processimages(readimages("testimages/"))    


def exec_on_imagelist(image,count):
	globals()['vertex%s' % count] = np.array( [[[125,eval(i).shape[0]],[450,325],[500,325],[875,eval(i).shape[0]]]], dtype=np.int32 )
    kernelsize=3




























def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
	
	
	
	
	
	
	
	
def readimages(dir):
    y = 0
    image=[]
    for i in os.listdir(dir):
        globals()['string%s' % y] = 'Hello'       
        i2=dir+i
        #image=mpimg.imread(i2)
        globals()['image%s' % y] = mpimg.imread(i2)
        y+=1
        #print('This image is:',i,type(image0), 'with dimensions:', image0.shape) 
    imagelist=[]
    for i in range(y):
        imagelist.append('image%s' % i)
    return(imagelist)    

def processimages(imagelist):
    yp=0
    for i in imagelist:
        print(i)
        plt.imshow(eval(i))
        globals()['imagegray%s' % yp] = grayscale(eval(i))
        yp+=1
    plt.imshow(imagegray1)
    
processimages(readimages("test_images/"))    
#print(imagelist)    


#plt.imshow(image5)

    #img2=np.copy(image)
    #img2lines = cv2.line(img2[300:,150:850],(24,99),(92,500),255)
    #plt.imshow(img2lines)

	
	
	
	
	
	
	