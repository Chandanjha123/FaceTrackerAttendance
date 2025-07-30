# Import required libraries
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier for face recognition
import cv2  # OpenCV for image capture and processing
import pickle  # To load pre-stored face data and labels
import numpy as np  # For numerical operations, reshaping, and flattening images
import os  # To check file paths (not used actively in this script)

# Load the Haar Cascade face detection model
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load label names from a pickle file (list of names corresponding to face data)
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

# Load flattened face data from a pickle file
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Convert list of faces into NumPy array
FACES = np.array(FACES)

# Validate shape and content of face data
if FACES.size == 0 or len(FACES.shape) != 2 or FACES.shape[1] != 7500:
    raise ValueError("⚠️ Loaded face data is empty or incorrectly shaped. Please re-collect data.")

# Create and train a KNN classifier with the loaded data
knn = KNeighborsClassifier(n_neighbors=5)  # Using 5 nearest neighbors
knn.fit(FACES, LABELS)  # Train model on face data and labels

# Function to open camera with specified index
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width of video frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height of video frame
    return cap

# Initialize first camera index
camera_index = 0
cap = open_camera(camera_index)  # Open the initial camera

# Display instructions to user
print("📷 Press 's' to switch camera | 'q' to quit")

# Start video stream loop
while True:
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"❌ Camera {camera_index} failed. Trying next...")
        cap.release()  # Release the failed camera
        camera_index = (camera_index + 1) % 5  # Try next camera (loop back after 4)
        cap = open_camera(camera_index)  # Open new camera
        continue

    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        continue  # Skip if frame not read properly

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]  # Crop the face region from frame
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize face to 50x50
        flattened_img = resized_img.flatten().reshape(1, -1)  # Flatten image into 1D array for prediction

        output = knn.predict(flattened_img)  # Predict label using trained KNN model

     # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle around the face

        # Prepare name text to display
        name_text = str(output[0])

        # Get text size for background box
        (text_width, text_height), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)

        # Set coordinates for red label box just above the green rectangle
        label_x1 = x
        label_y1 = y - text_height - 10  # 10px above the green box
        label_x2 = x + text_width + 10
        label_y2 = y

        # Ensure label stays within frame
        label_y1 = max(label_y1, 0)

        # Draw red filled rectangle above the face for the name label
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 255), -1)  # Red label box above face

        # Put white text (name) inside the red label box
        cv2.putText(frame, name_text, (label_x1 + 5, label_y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)  # White name text


    # Show the frame with drawings
    cv2.imshow(f"Camera {camera_index}", frame)

    key = cv2.waitKey(10) & 0xFF  # Wait for key press (with mask for compatibility)

    # Exit on pressing 'q'
    if key == ord('q'):
        print("👋 Exiting by 'q' key...")
        break

    # Switch camera on pressing 's'
    elif key == ord('s'):
        print(f"🔁 Switching camera {camera_index} → {(camera_index + 1) % 5}")
        cap.release()  # Release current camera
        cv2.destroyWindow(f"Camera {camera_index}")  # Close current window
        camera_index = (camera_index + 1) % 5  # Update camera index
        cap = open_camera(camera_index)  # Open next camera

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
