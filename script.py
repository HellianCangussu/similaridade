import cv2
from matplotlib import pyplot as plt


def orb():
    orb = cv2.ORB_create()

    image1 = cv2.imread('Imagem1.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('Imagem2.jpg', cv2.IMREAD_GRAYSCALE)
    
    kp_a = orb.detect(image1,None)
    kp_a, des_a = orb.compute(image1, kp_a)
    img1 = cv2.drawKeypoints(image1, kp_a, None, color=(0,255,0), flags=0)
    plt.imshow(img1), plt.show()  

    kp_b = orb.detect(image1,None)
    kp_b, des_b = orb.compute(image2, kp_a)
    img2 = cv2.drawKeypoints(image2, kp_b, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    similar_regions = [i for i in matches if i.distance < 50]
   
    if len(matches) == 0:
        return 0

    print("ORB:", len(similar_regions) / len(matches))
    # return len(similar_regions) / len(matches)


orb()
