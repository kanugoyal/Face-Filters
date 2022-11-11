import cv2
import matplotlib.pyplot as plt

def filter_prep(val):

    # For swag filter
    if val == 1:
        fil = cv2.imread('webapp/static/swag.jpg')

        return (fil)
    
    
    if val == 2:
        fil = cv2.imread('webapp/static/3dglasses.jpg')

        return (fil)
    
    
    if val == 3:
        fil = cv2.imread('webapp/static/eyes.jpg')

        return (fil)
    
    if val == 4:
        fil = cv2.imread('webapp/static/glasses.jpg')

        return (fil)

    if val == 5:
        fil = cv2.imread('webapp/static/spiderman.jpg')

        return (fil)
    
    if val == 6:
        fil = cv2.imread('webapp/static/ironman.png')

        return (fil)
    




