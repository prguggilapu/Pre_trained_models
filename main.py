### Let's start with loading ResNet50

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

resnet_model = ResNet50(weights='imagenet')

from tensorflow.keras.preprocessing import image

img_path = './images/dog.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = resnet_model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

"""### Let's run through a few test images"""

import cv2
from os import listdir
from os.path import isfile, join

# Our openCV function that displays the image and it's predicted labels
def draw_test(name, preditions, input_im):
    """Function displays the output of the prediction alongside the orignal image"""
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[1]+300 ,cv2.BORDER_CONSTANT,value=BLACK)
    img_width = input_im.shape[1]
    for (i,predition) in enumerate(preditions):
        string = str(predition[1]) + " " + str(predition[2])
        cv2.putText(expanded_image,str(name),(img_width + 50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
        cv2.putText(expanded_image,string,(img_width + 50,50+((i+1)*50)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    cv2.imshow(name, expanded_image)

# Get images located in ./images folder
mypath = "./images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Loop through images run them through our classifer
for file in file_names:

    from tensorflow.keras.preprocessing import image # Need to reload as opencv2 seems to have a conflict
    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    #load image using opencv
    img2 = cv2.imread(mypath+file)
    imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)

    # Get Predictions
    preds = resnet_model.predict(x)
    predictions = decode_predictions(preds, top=3)[0]
    draw_test("Predictions", predictions, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()

"""### Let's now load VGG16 and InceptionV3"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import vgg16, inception_v3, resnet50

#Loads the VGG16 model
vgg_model = vgg16.VGG16(weights='imagenet')

# Loads the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')

# Loads the ResNet50 model
# uncomment the line below if you didn't load resnet50 beforehand
#resnet_model = resnet50.ResNet50(weights='imagenet')

"""### Compare all 3 Models with the same test images"""

def getImage(path, dim=224, inception = False):
    img = image.load_img(path, target_size=(dim, dim))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if inception:
        x /= 255.
        x -= 0.5
        x *= 2.
    else:
        x = preprocess_input(x)
    return x

# Get images located in ./images folder
mypath = "./images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Loop through images run them through our classifer
for file in file_names:

    from tensorflow.keras.preprocessing import image # Need to reload as opencv2 seems to have a conflict
    #img = image.load_img(mypath+file, target_size=(dim, dim))
    x = getImage(mypath+file, 229)
    #load image using opencv
    img2 = cv2.imread(mypath+file)
    imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)

    # Get VGG16 Predictions
    x = getImage(mypath+file, 224)
    preds_vgg_model = vgg_model.predict(x)
    predictions_vgg = decode_predictions(preds_vgg_model, top=3)[0]
    draw_test("VGG16 Predictions", predictions_vgg, imageL)

    # Get Inception_V3 Predictions
    x = getImage(mypath+file, 299, inception = True)
    preds_inception = inception_model.predict(x)
    predictions_inception = decode_predictions(preds_inception, top=3)[0]
    draw_test("Inception_V3 Predictions", predictions_inception, imageL)

    # Get ResNet50 Predictions
    x = getImage(mypath+file, 224)
    preds_resnet = resnet_model.predict(x)
    predictions_resnet = decode_predictions(preds_resnet, top=3)[0]
    draw_test("ResNet50 Predictions", predictions_resnet, imageL)

    cv2.waitKey(0)

cv2.destroyAllWindows()