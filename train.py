import sklearn
import os
import cv2
from PIL import Image

logos = []
nologos = []

logodir = []

#ARD
ardlogodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/logo_ard'
ardnologodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/noLogo_ard'
ardMetadata = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/metadata_ard.txt'

#BRHD
brhdlogodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/logo_brHd'
brhdnologodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/noLogo_brHd'
brhdMetadata = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/metadata_brHd.txt'

#pro7
pro7logodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/logo_pro7'
pro7nologodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/noLogo_pro7'
pro7Metadata = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/metadata_pro7.txt'

#sat1
sat1logodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/logo_sat1'
sat1nologodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/noLogo_sat1'
sat1Metadata = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/metadata_sat1.txt'

#swr
swrlogodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/logo_swr'
swrnologodir='/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/noLogo_swr'
swrMetadata = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/metadata_swr.txt'



def cropImage(imagename, metadataFile):
    yourImage = Image.open(imagename)
    # <xMin>,<xMax>,<yMin>,<yMax>
    file = open(metadataFile , "r")
    text = file.read()
    text = text[0:-1]

    data = text.split(",", len(text))

    # img = yourImage.crop(())
    # xmin ymin xmax ymax
    xmin = int(data[1])
    xmax = xmin+int(data[2])
    ymin = int(data[3])
    ymax = ymin + int(data[4])

    img = yourImage.crop((xmin,ymin,xmax,ymax))
    return img

#ARD
logodir = os.fsencode(ardlogodir)
for file in os.listdir(logodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        logos.append(cropImage(ardlogodir+"/"+mystr, ardMetadata))


nologodir = os.fsencode(ardnologodir)
for file in os.listdir(nologodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        nologos.append(cropImage(ardnologodir+"/"+mystr , ardMetadata))
print("ARD FINISHED")
# brHd
logodir = os.fsencode(brhdlogodir)
for file in os.listdir(logodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        logos.append(cropImage(brhdlogodir + "/"+mystr, brhdMetadata))

nologodir = os.fsencode(brhdnologodir)
for file in os.listdir(nologodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        nologos.append(cropImage(brhdnologodir + "/"+mystr, brhdMetadata))
print("brhd FINISHED")
# pro7
logodir = os.fsencode(pro7logodir)
for file in os.listdir(logodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        logos.append(cropImage(pro7logodir + "/"+mystr, pro7Metadata))

nologodir = os.fsencode(pro7nologodir)
for file in os.listdir(nologodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        nologos.append(cropImage(pro7nologodir + "/"+mystr, pro7Metadata))
print("pro7 FINISHED")
# sat1
logodir = os.fsencode(sat1logodir)
for file in os.listdir(logodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        logos.append(cropImage(sat1logodir + "/"+mystr, sat1Metadata))

nologodir = os.fsencode(sat1nologodir)
for file in os.listdir(nologodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        nologos.append(cropImage(sat1nologodir + "/"+mystr, sat1Metadata))
print("sat1 FINISHED")
# swr
logodir = os.fsencode(swrlogodir)
for file in os.listdir(logodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        logos.append(cropImage(swrlogodir + "/"+mystr, swrMetadata))

nologodir = os.fsencode(swrnologodir)
for file in os.listdir(nologodir):
    # print (os.path.join(file))
    mystr = str(os.path.join(file))[2:-1]
    if mystr.endswith(".jpg"):
        nologos.append(cropImage(swrnologodir + "/"+mystr, swrMetadata))

print("swr FINISHED")




# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


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


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,
# 	help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())



# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images
# for (i, imagePath) in enumerate(imagePaths):
#     # load the image and extract the class label (assuming that our
#     # path as the format: /path/to/dataset/{class}.{image_num}.jpg
#     image = cv2.imread(imagePath)
#     label = imagePath.split(os.path.sep)[-1].split(".")[0]
#
#     # extract raw pixel intensity "features", followed by a color
#     # histogram to characterize the color distribution of the pixels
#     # in the image
#     pixels = image_to_feature_vector(image)
#     hist = extract_color_histogram(image)
#
#     # update the raw images, features, and labels matricies,
#     # respectively
#     rawImages.append(pixels)
#     features.append(hist)
#     labels.append(label)
#
#     # show an update every 1,000 images
#     if i > 0 and i % 1000 == 0:
#         print("[INFO] processed {}/{}".format(i, len(imagePaths)))


for logo in logos:

    # image = cv2.imread(logo)
    image = np.array(logo)
    label = "withlogos"
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

for nologo in nologos:
    # image = cv2.imread(nologo)
    image = np.array(nologo)

    label = "nologos"
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
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))


(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.25, random_state=42
)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.25, random_state=42
)

nneigbors = 10
model = KNeighborsClassifier(n_neighbors=nneigbors,n_jobs=1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# print("[INFO] evaluating histogram accuracy...")
# model2 = KNeighborsClassifier(n_neighbors=nneigbors,n_jobs=1)
# model2.fit(trainFeat, trainLabels)
# acc = model2.score(testFeat, testLabels)
# print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-img", "--imageToTest", required=True,
# 	help="path to image to test")
# ap.add_argument("-meta", "--metadataFile" , required )
# args = vars(ap.parse_args())



while(True):
    imageName = input("Give the image: ")
    metadataFile = input("Metadata file path: ")
    imageToPredict = cropImage(imageName + ".jpg" , metadataFile)
    testImage = np.array(imageToPredict)
    testPixels = image_to_feature_vector(testImage)
    predict1 = model.predict([testPixels])
    print(model.predict_proba([testPixels]))
    # if predict1 == "withLogo" or predict2 == "withLogo" :
    #     print("withLogo")
    print(model.predict([testPixels]))



