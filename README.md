# Fun face filters with Python Opencv

It uses Haar Cascade Classifiers, Dlib Face Landmark Detection and the Mediapipe fetures to detect mainly faces positions ,eye position, nose positions, face and hand landmark detection. It uses this information to overlay different accessories to the faces.

**Dlib face landmark detection dataset**
  shape_predictor_68_face_landmarks.dat
 
**HaarCascade applied**
  1. haarcascade_frontalface_default.xml
  2. haarcascade_mcs_nose.xml
  3. haarcascade_eye_tree_eyeglasses.xml
  4. haarcascade_smile.xml
  5. haarcascade_eye.xml


 **Filters**
 1. Swag Glasses
     python snap_filters/swagfilter.py
     image used : images/swag.png
  
 2. Pig Nose
     python snap_filters/nosefilter.py
     image used: images/pig_nose.png
    
 3. Fun Filter
     python snap_filters/funfilter.py
     image used: images/nose.png

 4. Capture Image With Smile
     python snap_filters/smile_selfie_cap.py
     Images stored in Selfie_Cap


**Virtual mouse**
 you'll get a mouse cursor that can move around, perform clicks and left / right / up / down swipes from your finger itself
 to run code:
 python vmouse.py

**Pressing keyboard keys using some object in webCam**
to run code: 
python useobject.py

**Difference b/w Haar Casacade Classifier and Dlib LIbrary**
   difference.txt

**To import file and run this code all the required modules are in requirement.txt**
    pip install -r requirements.txt
    


