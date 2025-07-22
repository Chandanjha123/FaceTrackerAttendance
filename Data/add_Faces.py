import cv2  # Import OpenCV for computer vision operations

# Function to open camera with specific index


# Load the Haar cascade file for face detection
faceDetect = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

faces_data=[]
i=0

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Open the camera using DirectShow backend (more stable for Windows)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)        # Set the width of the video frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)       # Set the height of the video frame
    return cap                                     # Return the VideoCapture object

camera_index = 0                   # Start with the first camera (usually the laptop cam)
cap = open_camera(camera_index)   # Open the first camera using our function

print("📷 Press 's' to switch camera | 'q' to quit")  # Instruction for the user

# Main loop to read and display frames
while True:
    if not cap.isOpened():  # Check if camera failed to open
        print(f"❌ Camera {camera_index} failed. Trying next...")
        cap.release()  # Release the broken camera
        camera_index = (camera_index + 1) % 5  # Go to next camera index (looping between 0–4)
        cap = open_camera(camera_index)  # Try to open the new camera
        continue  # Skip rest of loop and retry

    ret, frame = cap.read()  # Try reading a frame from the camera

    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert detected face  into gray scale
    faces=faceDetect.detectMultiScale(gray,1.3,5) # detectface using Gray Scale image


    for(x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w, :]
        resized_img=cv2.resize(crop_img,(50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame,str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Show the frame with rectangles (if any face is found)
    if not ret:  # If no frame was captured (invalid camera)
        print(f"⚠️ No frame from Camera {camera_index}. Trying next...")
        cap.release()  # Release camera resource
        camera_index = (camera_index + 1) % 5  # Try next camera
        cap = open_camera(camera_index)  # Open new camera
        continue  # Retry the loop

    # Display the frame on screen
    cv2.imshow(f"Camera {camera_index}", frame)  # Show frame with window title as camera index

    key = cv2.waitKey(1)  # Wait 1ms for a key press
    if key == ord('q') or len(faces_data)>100:  # If 'q' is pressed, exit the loop
        print("👋 Exiting...")
        break
    elif key == ord('s'):  # If 's' is pressed, switch to next camera
        print(f"🔁 Switching camera {camera_index} → {(camera_index + 1)%5}")
        cap.release()  # Release the current camera
        cv2.destroyAllWindows()  # Close the window of previous camera
        camera_index = (camera_index + 1) % 5  # Move to next camera
        cap = open_camera(camera_index)  # Open new camera

cap.release()  # Release the final camera after loop ends
cv2.destroyAllWindows()  # Close all OpenCV windows
