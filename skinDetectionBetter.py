import numpy as np
#np.set_printoptions(threshold=np.nan)
import cv2


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def ReadData():
    # Data in format [B G R Label] from
    data = np.genfromtxt('./Skin_NonSkin.txt', dtype=np.int32)

    # the first value in data array (:) gets all rows, the second value (3) gets the fourth column
    labels = data[:,3]
    # get all rows and the first 3 columns (at indexes 0,1,2)
    data = data[:,0:3]

    return data, labels


def TrainTree(data, labels):
    # data shape is one long array with 3 channels (245057,3)
    # need to shape into 3 number array for cvtColor function
    # bgr has shape (245057,1,3)  (rows, col, chan)
    bgr = np.reshape(data,(data.shape[0],1,3))
    hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    # once we have converted to HSV, we reshape back to original shape of (245057,3)
    dataHSV = np.reshape(hsv,(hsv.shape[0],3))

    trainData, testData, trainLabels, testLabels = train_test_split(dataHSV, labels, test_size=0.20, random_state=42)

    print(trainData.shape)
    print(trainLabels.shape)
    print(testData.shape)
    print(testLabels.shape)

    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # Initialize our classifier
    gnb = GaussianNB()
    clf = gnb.fit(trainData, trainLabels)

    return clf


# first let's train the data
data, labels = ReadData()
classifier = TrainTree(data, labels)

# get the webcam. The input is either a video file or the camera number
# since using laptop webcam (only 1 cam), input is 0. A 2nd cam would be input 1
camera = cv2.VideoCapture(0)

while True:
    # reads in the current frame
    # .read() returns True if frame read correctly, and False otherwise
    ret, frame = camera.read()

    if ret:
        # reshape the frame to follow format of training data (rows*col, 3)
        data = np.reshape(frame, (frame.shape[0] * frame.shape[1], 3))
        bgr = np.reshape(data, (data.shape[0], 1, 3))
        hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
        # once we have converted to HSV, we reshape back to original shape of (245057,3)
        data = np.reshape(hsv, (hsv.shape[0], 3))
        predictedLabels = classifier.predict(data)

        # predictedLabels consists of 1 (skin) and 2 (non-skin), needs to change to 0 (non-skin) and 255 (skin)
        predictedMask = (-(predictedLabels - 1) + 1) * 255

        # resize to match frame shape
        imgLabels = np.zeros(frame.shape, dtype="uint8")
        imgLabels = np.reshape(predictedLabels,(frame.shape[0],frame.shape[1]))
        imgLabels = ((-(imgLabels-1)+1)*255)
        # do bitwsie AND to pull out skin pixels. All skin pixels are anded with 255 and all others are 0
        cv2.imwrite('./resultNew.png', imgLabels)
        imageMask = cv2.imread("./resultNew.png")
        # masks require 1 channel, not 3, so change from BGR to GRAYSCALE
        imgLabels = cv2.cvtColor(imageMask, cv2.COLOR_BGR2GRAY)
        skin = cv2.bitwise_and(frame, frame, mask=imgLabels)
        #print(skin.shape)  # (480,640,3)

        #print(predictedMask.shape) # (307200,)
        #print(data.shape)  # (307200,3)
        #print(predictedLabels)  # [2 2 2 1 1 2 2 1 ...]
        #print(predictedLabels.shape)  # (480,640,3)
        #print(imgLabels.shape)  # (480,640)

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
