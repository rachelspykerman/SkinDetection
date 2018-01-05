# import the necessary packages
import numpy as np
import cv2



# define the lower and upper boundaries for detecting the HSV skin color
#lower = np.array([0, 48, 80], dtype = "uint8")
lower = np.array([0, 10, 20], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")


# get the webcam. The input is either a video file or the camera number
# since using laptop webcam (only 1 cam), input is 0. A 2nd cam would be input 1
camera = cv2.VideoCapture(0)

while(True):
    # reads in the current frame
    # .read() returns True if fram read correctly, and False otherwise
    ret, frame = camera.read()

    if ret == True:
        # convert frame to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the specified upper and lower boundaries
        # use HSV since RGB is very sensitive to illumination changes
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Pixels that are white (255) in the mask represent areas of the frame that are skin pixels
        # Pixels that are black (0) in the mask represent non-skin pixels
        # skinMask will only contain 1's for skin pixels and 0's for non-skin pixels
        skinMask = cv2.inRange(converted, lower, upper)


        # Create an elliptical structuring kernel which is then used to perform two iterations
        # of erosions and dilations on the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # Erosions and dilations will help remove the small false-positive skin regions in the image
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # Blur the mask to help remove noise, then apply the mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        # the AND operator applies the skinMask to the image
        print("++++++++++++++++++++++++++++++")
        print(skinMask)
        print(skinMask.shape)  # (480, 640)
        print(frame.shape)  # (480,640,3)
        print("++++++++++++++++++++++++++++++")
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # show the skin in the image along with the mask, show images side-by-side
        cv2.imshow("images", np.hstack([frame, skin]))

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


# release the video capture
camera.release()
cv2.destroyAllWindows()

# Notes
# Skin detection is difficult due to:
#    1. Illumination: changes in light source and illumination (indoor, outdoor, shadows, non-white lights)
#    2. Camera characteristics: even under same illumination, skin-color changes between cameras due to diff
#       camera sensor characteristics
#    3. Ethnicity: skin color changes from person to person
#    4. Individual characteristics: age, sex, body parts also affect skin color
#    5. Other Factors: subject appearance (makeup, hair, glasses), background color, shadows, motion

