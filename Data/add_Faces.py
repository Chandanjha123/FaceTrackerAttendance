import cv2  # Import OpenCV library for image processing and face detection
import pickle  # Import pickle for saving and loading Python objects (like lists, arrays)
import numpy as np  # Import NumPy for handling arrays and numerical operations
import os  # Import OS module for interacting with the file system
import sys  # Import sys for accessing command-line arguments

# 📌 FIX: Build full absolute path to cascade XML file
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')  # Resolve correct path
faceDetect = cv2.CascadeClassifier(cascade_path)  # Load the Haar Cascade face detection model

# ✅ Safety check to make sure the classifier was loaded
if faceDetect.empty():
    raise IOError(f"❌ Failed to load Haar cascade from: {cascade_path}")

faces_data = []  # Initialize an empty list to store collected face images

name = input("Enter your name:")  # Prompt user to enter their name

i = 0  # Initialize frame counter used to control the frequency of saving images

# Define a function to open a camera using its index
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Open camera at given index using DirectShow backend (Windows-specific)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set camera frame width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set camera frame height to 480
    return cap  # Return the camera object

camera_index = 0  # Start with camera index 0
cap = open_camera(camera_index)  # Open the default camera

print("📷 Press 's' to switch camera | 'q' to quit")  # Display usage instructions

# Start capturing frames in a loop
while True:
    if not cap.isOpened():  # Check if the camera failed to open
        print(f"❌ Camera {camera_index} failed. Trying next...")  # Notify the user
        cap.release()  # Release the current camera
        camera_index = (camera_index + 1) % 5  # Try the next camera index (modulo to loop back)
        cap = open_camera(camera_index)  # Try to open the new camera
        continue  # Skip rest of the loop and retry with the new camera

    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:  # If frame capture failed
        continue  # Skip this iteration and continue to the next

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image

    for (x, y, w, h) in faces:  # Loop through all detected faces
        crop_img = frame[y:y+h, x:x+w, :]  # Crop the face from the frame
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize the cropped face to 50x50 pixels

        if len(faces_data) < 100 and i % 10 == 0:  # Save one image every 10 frames until 100 samples are collected
            faces_data.append(resized_img)  # Append the resized face image to the dataset
        i += 1  # Increment the frame counter

        # Display number of collected samples on screen
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with drawings in a window
    cv2.imshow(f"Camera {camera_index}", frame)

    key = cv2.waitKey(10) & 0xFF  # Wait for a key press for 10ms

    if key == ord('q'):  # If 'q' is pressed
        print("👋 Exiting by 'q' key...")  # Notify the user
        break  # Exit the loop
    elif key == ord('s'):  # If 's' is pressed
        print(f"🔁 Switching camera {camera_index} → {(camera_index + 1) % 5}")  # Notify camera switch
        cap.release()  # Release current camera
        cv2.destroyWindow(f"Camera {camera_index}")  # Close the current camera window
        camera_index = (camera_index + 1) % 5  # Increment and wrap camera index
        cap = open_camera(camera_index)  # Open the new camera
    elif len(faces_data) == 100:  # If 100 samples are collected
        print("✅ Collected 100 face samples. Exiting...")  # Notify completion
        break  # Exit the loop

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Convert list of face images to a NumPy array
faces_data = np.asarray(faces_data)
# Flatten each 50x50x3 image to a 1D array (shape becomes: 100 x 7500)
faces_data = faces_data.reshape(faces_data.shape[0], -1)

# Make sure the 'data' folder exists, if not, create it
os.makedirs("data", exist_ok=True)

# If names.pkl doesn't exist, create it and save names list
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * len(faces_data)  # Create a list of the same name repeated
    with open('data/names.pkl', 'wb') as f:  # Open file in write-binary mode
        pickle.dump(names, f)  # Save the list to file
else:
    with open('data/names.pkl', 'rb') as f:  # Open existing names.pkl
        names = pickle.load(f)  # Load existing names list
    names += [name] * len(faces_data)  # Append new names to the list
    with open('data/names.pkl', 'wb') as f:  # Open the file again for writing
        pickle.dump(names, f)  # Save updated list back to file

# If faces_data.pkl doesn't exist, create it and save face data
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:  # Open file in write mode
        pickle.dump(faces_data, f)  # Save face data array
else:
    with open('data/faces_data.pkl', 'rb') as f:  # Open existing face data file
        existing_faces = pickle.load(f)  # Load existing face data

    # Ensure the new and existing data have the same shape before appending
    if existing_faces.shape[1] != faces_data.shape[1]:
        print("⚠️ Shape mismatch! Skipping append.")  # Warn if shape doesn't match
    else:
        # Append new face data to existing face data
        combined_faces = np.append(existing_faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:  # Open file for writing
            pickle.dump(combined_faces, f)  # Save the combined dataset
