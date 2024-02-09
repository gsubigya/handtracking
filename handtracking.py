import cv2
import mediapipe as mp

# **Initialize MediaPipe for hand detection and drawing utilities**
mpHands = mp.solutions.hands
Hands = mpHands.Hands()  # Initialize the MediaPipe Hands solution
mpDraw = mp.solutions.drawing_utils  # Tools for drawing landmarks/connections

# **Open the default camera**
cap = cv2.VideoCapture(0)  # 0 usually indicates the default webcam

# **Error handling if the camera fails to open**
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# **Main processing loop**
while True:
    # **Read a frame from the camera**
    success, img = cap.read()

    # **Check if the frame was read successfully**
    if not success:
        print("Error reading video stream")
        break

    # **Convert from BGR (OpenCV default) to RGB (MediaPipe requirement)**
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # **Process the image using MediaPipe to detect hands**
    results = Hands.process(converted_image)

    # **If hands are detected, draw landmarks**
    if results.multi_hand_landmarks:
        for hand_in_frame in results.multi_hand_landmarks:
            # **Draw hand landmarks and connections on the frame**
            mpDraw.draw_landmarks(img, hand_in_frame, mpHands.HAND_CONNECTIONS)

    # **Display the frame with hand landmarks**
    cv2.imshow("Hand Tracking", img)

    # **Exit the loop if the 'q' key is pressed**
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# **Clean up after finishing**
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close the display window
