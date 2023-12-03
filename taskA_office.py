# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:42 2023

@author: User
"""

import cv2
import numpy as np

# Read video and image 
vid = cv2.VideoCapture("office.mp4") 
overlay_video = cv2.VideoCapture("talking.mp4")
end_screen = cv2.VideoCapture("endscreen.mp4")
watermark1 = cv2.imread("watermark1.png",1)
watermark2 = cv2.imread("watermark2.png",1)

# Set the file name of the new video, codec, frame rate, resolution (width, height)
output = cv2.VideoWriter("office_processed.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280,720))

total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT) # Get total number of frames
fps = vid.get(cv2.CAP_PROP_FPS) # Get fps

# Threshold value to detect day/night
threshold = 100
night_detected = False

# Detect face
face_cascade = cv2.CascadeClassifier("face_detector.xml")  # Load pre-trained Haar cascade model

# Resize the overlay video to a smaller size
resized_overlay_width = 270
resized_overlay_height = 150

# Watermark images array
watermarks = [watermark1, watermark2]
masks = []

# Extract the black for each watermark
for watermark in watermarks:
    gray_mark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    [nrow, ncol] = gray_mark.shape
    mask = np.zeros((nrow, ncol), dtype=np.uint8)
    for x in range(0, nrow):
        for y in range(0, ncol):
            if gray_mark[x, y] <= 100:
                mask[x, y] = 255
            else:
                mask[x, y] = 0
    masks.append(mask)

# Decide which watermark and mask to use
mark_type = watermark1   
mask = masks[0]   

for frame_count in range(0, int(total_no_frames)): # To loop through all the frames

    # Success - boolean value that indicates whether the frame was successfully read
    success, frame = vid.read() # Read a single frame from the video

    if not success:
        break  # No more frames to read, exit the loop

    # Detect night or day based on first frame
    if frame_count == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_bright = gray_frame.mean()

        frame = frame.astype(np.float64)

        if avg_bright < threshold:
            night_detected = True

    if night_detected:
            frame = frame * 1.5 + 20

    # Ensure values are within the valid range
    frame[frame < 0] = 0  # Set negative values to 0
    frame[frame > 255] = 255  # Set values over 255 to 255
    frame = frame.astype(np.uint8)


    # Detect face
    faces = face_cascade.detectMultiScale(frame, 1.3, 5) # Perform face detection.
    for (x, y, w, h) in faces: # To loop through all the detected faces.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw a bounding box

        face_roi = frame[y:y + h, x:x + w]  # Extract the region of interest (ROI) for each detected face
        blurred_face = cv2.GaussianBlur(face_roi, (35, 35), 0) # Apply Gaussian blur to the face region
        frame[y:y + h, x:x + w] = blurred_face # Replace the original face region with the blurred one


    # Overlay Video
    success_overlay, frame_overlay = overlay_video.read()

    if not success_overlay:
        overlay_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning if the overlay video ends

    # Create a border around the overlay frame
    frame_overlay = cv2.copyMakeBorder(frame_overlay, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Resize the overlay frame
    frame_overlay = cv2.resize(frame_overlay, (resized_overlay_width, resized_overlay_height))

    # Ensure frame_overlay has three channels
    if len(frame_overlay.shape) == 2:  # If it's a grayscale image
        frame_overlay = cv2.cvtColor(frame_overlay, cv2.COLOR_GRAY2BGR)  

    # Define the position to place the overlay 
    y_offset = 80
    x_offset = 80

    # Overlay the resized talking video on the main video
    frame[y_offset:y_offset+resized_overlay_height, x_offset:x_offset+resized_overlay_width] = frame_overlay


    # Watermark Application
    current_time = frame_count / fps  - 1 # Check the current time to decide which watermark to apply

    # If current time can be divided by 5 then switch watermark
    if current_time != 0 and current_time % 5 == 0: 
        if mark_type is watermark1:
            mark_type = watermark2
            mask = masks[1] 
        else:
            mark_type = watermark1
            mask = masks[0] 


    # Get the image without the area where the watermark will be placed
    background = cv2.bitwise_and(frame, frame, mask = mask)
    # Invert the mask so that the text is the ROI
    mask = cv2.bitwise_not(mask)
    # Get the text
    text = cv2.bitwise_and(mark_type, mark_type, mask=mask)
    # Add watermark
    frame = cv2.add(background, text)    
    # Invert the mask back to original
    mask = cv2.bitwise_not(mask)

    output.write(frame) # Save the processed frame to output variable


# Concatenate end screen video
while True:
    success_end, frame2 = end_screen.read()

    if not success_end:
        break  # Break the loop if the end screen video ends

    output.write(frame2)


# Ensures file is properly closed and saved
output.release()
vid.release()
overlay_video.release()
cv2.destroyAllWindows()