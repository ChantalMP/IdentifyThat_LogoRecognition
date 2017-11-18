import cv2
import numpy as np
import os

# requires a name of an image,that is in jpg format.(the image must be at the main folder of this project, else, picturedir can be changed to match the folder.
def findAndIdentify(imageName):

    templatesDir = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/templates'
    pictureDir = '/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum'
    try:
        # imgToSearch = cv2.imread(pictureDir + '/' + imageName + '.jpg')
        imgToSearch = cv2.imread(imageName)
    except:
        print("no images found")
        return

    img_grey = cv2.cvtColor(imgToSearch, cv2.COLOR_BGR2GRAY)

    locs = []

    for file in os.listdir(templatesDir):
        if not file.endswith(".jpg"):
            continue
        template = cv2.imread(templatesDir + "/"+ file, 0)

        res = cv2.matchTemplate(img_grey, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.75
        w, h= template.shape[::-1]
        shape = (w,h)

        loc = np.where(res >= threshold)
        if(len(loc[0]) is not 0):
            prob = 0
            for y in res:
                for x in y:
                    if (x >= threshold):
                        prob = max(prob , x)

            locs.append([loc, shape, file[:-4], prob])

    probs = []
    for i in range(0, len(locs)):
        probs.append(locs[i][3])

    if len(probs) is not 0:
        index = probs.index(max(probs))
        w,h = locs[index][1]
        for pt in zip(*locs[index][0][::-1]):
            cv2.rectangle(imgToSearch, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        # print gewinner
        print(locs[index][2])
        # cv2.imshow('detected', imgToSearch)
        return (imgToSearch, locs[index][2])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        return (imgToSearch , "No Logo")
        print("no logo")

# while(True):
#     imageName = input("Give the image: ")
#     findAndIdentify(imageName)