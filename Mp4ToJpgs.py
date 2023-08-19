import numpy as np
import cv2
import os



# Specify the path for your video file
video_path = 'C:/Users/eran/Pictures/Saved Pictures/23.mp4'

# Specify the directory to save frames
save_path = 'C:/Users/eran/Pictures/Saved Pictures/jpegs'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if opened successfully
if not cap.isOpened(): 
    print("Error: Cannot open video.")
else:
    # Get video meta-data
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Frame count: {frame_count}, FPS: {fps}")
    
    count = 2000
    success = True

    # Loop until end of video
    while success:
        success, frame = cap.read()
        if success:
            # Save frame as JPG file
            save_name = os.path.join(save_path, f"frame{count:04d}.jpg")
            cv2.imwrite(save_name, frame)
            print(f'Successfully wrote frame {count:04d}')
            count += 1

    print("Completed!")
    
# Release the video capture object
cap.release()