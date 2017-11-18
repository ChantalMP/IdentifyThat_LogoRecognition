import sklearn
import os
import cv2
from PIL import Image
import  os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

hackDir = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/'

withLogos = []
noLogos = []



def cropImage(imagename, metadataFile, factor=10):
    yourImage = Image.open(imagename)
    w,h = yourImage.size
    # <xMin>,<xMax>,<yMin>,<yMax>
    # <xmin>, <ymin> , <xmax> , <yMax>
    with open(metadataFile , "r") as file:

        text = file.read()
        text = text[0:-1]

        data = text.split(",", len(text))

        # img = yourImage.crop(())
        # xmin ymin xmax ymax
        xmin = 0 if int(data[1]) - factor < 0 else int(data[1]) - factor
        xmax = w if xmin+int(data[3]) + factor+factor > w else xmin+int(data[3]) + factor+factor
        ymin = 0 if int(data[2]) - factor < 0 else int(data[2]) - factor
        ymax = h if ymin + int(data[4]) + factor+factor > h else ymin + int(data[4]) + factor+factor
        # print(xmin,xmax,ymin,ymax)

        img = yourImage.crop((xmin,ymin,xmax,ymax))
        img = img.resize((int(w/2),int(h/2)))
        return img


def detectLogo(image):
    w,h = image.size
    print(w,h)
    for y in range(0,102 , step=15):
        for x in range(0,720 , step= 15):




for subdir in os.listdir(hackDir + 'Images'):

    fileDir = hackDir + 'Images/' + subdir
    if not os.path.isfile(fileDir + "/metadata.txt"):
        continue

    metadata = fileDir + "/metadata.txt"

    if os.path.isdir(fileDir):
        for file in os.listdir(fileDir):
            if file.endswith(".jpg"):
                if subdir[0:2] == "no":
                    noLogos.append(cropImage(fileDir + "/" + file, metadata))
                else:
                    withLogos.append(cropImage(fileDir + "/" + file, metadata))
        print(subdir + ' finished')




def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

for logo in withLogos:
    image = np.array(logo)
    label = "withlogo"
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

for nologo in noLogos:
    image = np.array(nologo)
    label = "nologo"
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
# training raw pixel
(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.25, random_state=42
)

nneigbors = 1
model = KNeighborsClassifier(n_neighbors=nneigbors,n_jobs=1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# # training histogram
# (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
#     features, labels, test_size=0.25, random_state=42
# )


# print("[INFO] evaluating histogram accuracy...")
# model2 = KNeighborsClassifier(n_neighbors=nneigbors,n_jobs=1)
# model2.fit(trainFeat, trainLabels)
# acc = model2.score(testFeat, testLabels)
# print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


# construct the argument parse and parse the arguments over command line
# ap = argparse.ArgumentParser()
# ap.add_argument("-img", "--imageToTest", required=True,
# 	help="path to image to test")
# ap.add_argument("-meta", "--metadataFile" , required )
# args = vars(ap.parse_args())



while(True):
    imageName = input("Give the image: ")

    metadataFile = input("Metadata file path: ")
    try:
        imageToPredict = cropImage(imageName + ".jpg" , metadataFile + ".txt")
        detectLogo(imageToPredict)
        # imageToPredict = Image.open(imageName + ".jpg")
        # w, h = imageToPredict.size
        # imageToPredict = imageToPredict.resize((int(w / 2), int(h / 2)))
    except:
        print("file not found")
        continue

    testImage = np.array(imageToPredict)
    testPixels = image_to_feature_vector(testImage)
    predict1 = model.predict([testPixels])
    print(model.predict_proba([testPixels]))
    print(model.predict([testPixels]))



