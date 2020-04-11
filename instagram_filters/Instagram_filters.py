import cv2
import numpy as np


low_sigmaS_edgePreservingFilter = 0
high_sigmaS_edgePreservingFilter = 20
sigmaS_edgePreservingFilter = 2

low_sigmaR_edgePreservingFilter = 0
high_sigmaR_edgePreservingFilter = 20
sigmaR_edgePreservingFilter = 0.07

low_sigmaS_detailEnhance = 0
high_sigmaS_detailEnhance = 10
sigmaS_detailEnhance = 3

low_sigmaR_detailEnhance = 0
high_sigmaR_detailEnhance = 20
sigmaR_detailEnhance = 0.25

blurAmount = 10
maxBlurAmount = 40

# Function for all trackbar calls
def cartoonify():
    global image

    image_copy = image.copy()

    image_copy = cv2.edgePreservingFilter(image_copy, flags=1, sigma_s=sigmaS_edgePreservingFilter, sigma_r=sigmaR_edgePreservingFilter)

    image_copy = cv2.detailEnhance(image_copy, sigma_s=sigmaS_detailEnhance, sigma_r=sigmaR_detailEnhance)

    grayImage = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)


    grayImageInv = 255 - grayImage.copy()

    if(blurAmount > 0):
        gaussian = cv2.GaussianBlur(grayImageInv,
                        (2 * blurAmount + 1, 2 * blurAmount + 1), 0)
    else:
        gaussian = grayImageInv

    # gaussian = cv2.GaussianBlur(grayImageInv, (47, 47), 0)
    blend = cv2.divide(grayImage, 255-gaussian, scale=256.0)

    # Sharpen kernel
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    # Using 2D filter by applying the sharpening kernel
    sharpenOutput = cv2.filter2D(blend, -1, sharpen)

    print("sigmaS_edgePreservingFilter: ", sigmaS_edgePreservingFilter)
    print("sigmaR_edgePreservingFilter: ", sigmaR_edgePreservingFilter)
    print("sigmaS_detailEnhance: ", sigmaS_detailEnhance)
    print("sigmaR_detailEnhance: ", sigmaR_detailEnhance)
    print("blurAmount: ", 2*blurAmount+1)

    cv2.imshow("Cartoonify",sharpenOutput)
    k = cv2.waitKey(0)

    if k==27:
        cv2.destroyAllWindows()

# Function to update sigmaS_edgePreservingFilter
def update_sigmaS_edgePreservingFilter( *args ):
    global sigmaS_edgePreservingFilter
    sigmaS_edgePreservingFilter = args[0]/10
    cartoonify()
    pass

# Function to update sigmaR_edgePreservingFilter
def update_sigmaR_edgePreservingFilter( *args ):
    global sigmaR_edgePreservingFilter
    sigmaR_edgePreservingFilter = args[0]/100
    cartoonify()
    pass

# Function to update sigmaS_detailEnhance
def update_sigmaS_detailEnhance( *args ):
    global sigmaS_detailEnhance
    sigmaS_detailEnhance = args[0]
    cartoonify()
    pass

# Function to update sigmaR_detailEnhance
def update_sigmaR_detailEnhance( *args ):
    global sigmaR_detailEnhance
    sigmaR_detailEnhance = args[0]/50
    cartoonify()
    pass

# Function to update blur amount
def updateBlurAmount( *args ):
    global blurAmount
    blurAmount = args[0]
    cartoonify()
    pass


# Read sample image
image = cv2.imread('trump.jpg')

image_copy = image.copy()
# Display images
cv2.namedWindow("Cartoonify", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Cartoonify",image_copy)

# Trackbar to control the sigmaS_edgePreservingFilter
cv2.createTrackbar( "sigmaS edgePreservingFilter", "Cartoonify", low_sigmaS_edgePreservingFilter,
            high_sigmaS_edgePreservingFilter, update_sigmaS_edgePreservingFilter)

# Trackbar to control the sigmaR_edgePreservingFilter
cv2.createTrackbar( "sigmaR edgePreservingFilter", "Cartoonify", low_sigmaR_edgePreservingFilter,
            high_sigmaR_edgePreservingFilter, update_sigmaR_edgePreservingFilter)

# Trackbar to control the sigmaS_edgePreservingFilter
cv2.createTrackbar( "sigmaS detailEnhance", "Cartoonify", low_sigmaS_detailEnhance,
            high_sigmaS_detailEnhance, update_sigmaS_detailEnhance)

# Trackbar to control the sigmaS_edgePreservingFilter
cv2.createTrackbar( "sigmaR detailEnhance", "Cartoonify", low_sigmaR_detailEnhance,
            high_sigmaR_detailEnhance, update_sigmaR_detailEnhance)

# Trackbar to control the blur
cv2.createTrackbar( "Blur", "Cartoonify", blurAmount, maxBlurAmount,
            updateBlurAmount)

k = cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()
