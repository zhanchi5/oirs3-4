import cv2
import mahotas
import pdb
bins = 8

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# feature descriptor 4: SIFT
def fd_4(image, mask=None):
    imag = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    imag=cv2.drawKeypoints(gray,kp,imag)
    cv2.normalize(imag, imag)
    return imag.flatten()


# feautre descriptor 5: SURF
def fd_5(image, mask=None):
    #pdb.set_trace()
    # img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints_surf, descriptros = surf.detectAndCompute(image, None)
    img = cv2.drawKeypoints(image, keypoints_surf, None)
    cv2.normalize(img, img)
    return img.flatten()


# feature descriptor 6: ORB
def fd_6(image, mask=None):
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints_orb, descriptors = orb.detectAndCompute(image, None)
    img = cv2.drawKeypoints(image, keypoints_orb, None)
    cv2.normalize(img,img)
    return img.flatten()