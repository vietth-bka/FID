__author__ = "vietth5, datnt527"

import cv2

def is_blur(img):
    """Decide if the input image is blur or not
    :param img: input image
    :type img: numpy.array
    :returns True if image is blur, return blury as well
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(img, cv2.CV_64F)
    blury = laplace.var()

    # the more blur, the smaller output
    if blury <= 50.:
        # This image is blur ==> need to remove!
        return True, blury
    else:
        # 'This image has good quality ==> retain!'
        return False, blury
