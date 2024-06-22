```diff
-def apply_post_processing(prediction):
     Convert the prediction to a binary mask using a threshold.
     Convert the continuos values in the prediction into discrete 1s or 0s.
     If prediction thresh is more than 0.3 it`s considered as 1 (True).
     Otherwise a 0 (false)
     Binary mask means define the region of interest of the image
-    binary_mask = (prediction > 0.3).astype(np.uint8)

-    if np.any(binary_mask):  # Check if the mask is not empty
         Ensure that the binary mask is a NumPy array
-        binary_mask = np.asarray(binary_mask, dtype=np.uint8)

         Erosion operation to remove small false positives.
         Iteration represents, how many time you moved the default kernal
         across the image matrix. Erosion operation will reduce the noise
         and let the important feature emerged more strongly.
-        eroded_mask = cv2.erode(binary_mask, erosion_kernel, iterations=8)

         Dilation operation to restore the size of the image and readjust the too much impacted regions during the erosion process
-        post_processed_mask = cv2.dilate(eroded_mask, erosion_kernel, iterations=5)

-        return post_processed_mask
-    else:
-        return binary_mask

-import cv2
-import numpy as np
-from tensorflow.keras.models import load_model
-from tensorflow.keras.preprocessing import image
-from tensorflow.keras.applications.vgg16 import preprocess_input
-from scipy.ndimage import binary_erosion, binary_dilation

 Load your pre-trained VGG16 model for gun holding
-vgg_model = load_model('custom_gun_model.h5')

 Create a VideoCapture object
-cap = cv2.VideoCapture(0)

 Set the width and height of the video capture
 3- means width. Set the pixel density for width.
-cap.set(3, 640)
 4- means height. Set the pixel density for height.
-cap.set(4, 480)

 Initialize variables for temporal smoothing
-history_length = 5
-prediction_history = []

 Initialize a binary erosion kernel with size of 5 x 5. This is a smoothing technique used to reduce the noise, as averages the pixel in the overlapping region of the kernel.
-erosion_kernel = np.ones((5, 5), np.uint8)

-while True:
     Capture a frame from the video feed
-    ret, frame = cap.read()

     Check if the frame is not empty
-    if not ret:
-        print("Error: Could not read frame")
-        break

     Preprocess the frame for VGG16 model
-    frame_vgg = cv2.resize(frame, (224, 224))
-    img_array = image.img_to_array(frame_vgg)
-    img_array = np.expand_dims(img_array, axis=0)
-    img_array = preprocess_input(img_array)

     Fine-tune the model on your specific gun detection task.
     2 zeros are referring to like 2 dimensional representation of the prediction
-    prediction = vgg_model.predict(img_array)[0][0]

     Append the current prediction to the history
-    prediction_history.append(prediction)

     Keep only the last 'history length' predictions
-    if len(prediction_history) > history_length:
-        prediction_history = prediction_history[1:]
        
     camera feed is a live feed subject to variations. History prediction is maintained as a moving average strategy to overcome the variations which could happen in the camera`s live feed and to make the prediction more stabalized.

     Calculate the smoothed prediction using a simple moving average
-    smoothed_prediction = np.mean(prediction_history)

     Apply post-processing to the prediction
-    post_processed_prediction = apply_post_processing(smoothed_prediction)

     Assuming binary classification (gun or not)
-    prediction_label = "Gun Detected" if post_processed_prediction > 0.5 else "No Gun Detected"

     Display the frame with smoothed prediction
-    cv2.putText(frame, f"Prediction: {prediction_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
-    cv2.imshow('Gun Detection', frame)
-    cv2.waitKey(0)

     Break the loop when 'q' key is pressed
-    if cv2.waitKey(1) & 0xFF == ord('q'):
-        break

 Release the Video Capture and close all windows
-cap.release()
-cv2.destroyAllWindows()
```
**Output :- In here the output is on live feedback. Imaging, when user act as hang on the gun in front of the web camera, so in that time the model is detected as that person is the bugler or thief who have a gun.**
