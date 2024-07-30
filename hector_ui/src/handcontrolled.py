
# code1.py
import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import ttk
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import rospy

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to initialize camera
def init_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera with index {index}")
        return None
    # Set camera properties to ensure color feed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# Initialize ROS node
rospy.init_node('HectorQ_GUI', anonymous=False)

# Publishers
takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1)
land_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=1)
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

# Send UAV control commands
def send_uav_command(command):
    if command == "takeoff":
        takeoff_pub.publish(Empty())
    elif command == "land":
        land_pub.publish(Empty())
    else:
        vel_msg = Twist()
        if command == "go_up":
            vel_msg.linear.z = 1.0
        elif command == "go_down":
            vel_msg.linear.z = -1.0
        elif command == "go_forward":
            vel_msg.linear.x = 1.0
        elif command == "go_backward":
            vel_msg.linear.x = -1.0
        elif command == "go_left":
            vel_msg.linear.y = 1.0
        elif command == "go_right":
            vel_msg.linear.y = -1.0
        elif command == "turn_clockwise":
            vel_msg.angular.z = -1.0
        elif command == "turn_counter_clockwise":
            vel_msg.angular.z = 1.0
        vel_pub.publish(vel_msg)

# Function to determine gesture based on hand position relative to center and edges
def determine_gesture(hand_landmarks, center_point, frame_width, frame_height):
    wrist = hand_landmarks.landmark[0]
    wrist_point = (int(wrist.x * frame_width), int(wrist.y * frame_height))
    
    # Define edge thresholds (as a fraction of the frame width/height)
    edge_threshold = 0.1  # 10% of the width/height

    # Define edge boundaries
    left_edge = frame_width * edge_threshold
    right_edge = frame_width * (1 - edge_threshold)
    top_edge = frame_height * edge_threshold
    bottom_edge = frame_height * (1 - edge_threshold)

    # Debugging statements
    print(f"Wrist Position: {wrist_point}")
    print(f"Top Edge: {top_edge}, Bottom Edge: {bottom_edge}")
    print(f"Left Edge: {left_edge}, Right Edge: {right_edge}")

    # Determine gesture based on hand position
    if wrist_point[1] < top_edge:
        # Hand is near the top edge
        if wrist_point[0] < left_edge:
            return "turn_counter_clockwise"
        elif wrist_point[0] > right_edge:
            return "turn_clockwise"
        else:
            return "go_up"
    elif wrist_point[1] > bottom_edge:
        # Hand is near the bottom edge
        if wrist_point[0] < left_edge:
            return "go_left"
        elif wrist_point[0] > right_edge:
            return "go_right"
        else:
            return "go_down"
    elif wrist_point[1] < center_point[1] and wrist_point[1] >= top_edge:
        # Hand is between the top edge and center
        if wrist_point[0] < left_edge:
            return "turn_counter_clockwise"
        elif wrist_point[0] > right_edge:
            return "turn_clockwise"
        else:
            return "go_forward"
    elif wrist_point[1] > center_point[1] and wrist_point[1] <= bottom_edge:
        # Hand is between the bottom edge and center
        if wrist_point[0] < left_edge:
            return "go_left"
        elif wrist_point[0] > right_edge:
            return "go_right"
        else:
            return "go_backward"
    else:
        # Hand is near the center
        if wrist_point[0] < left_edge:
            return "go_left"
        elif wrist_point[0] > right_edge:
            return "go_right"
        else:
            return "hover"

# Try different camera indices
cap = None
for i in range(5):
    cap = init_camera(i)
    if cap:
        break

if not cap:
    print("Failed to open camera. Exiting...")
    exit()

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(frame_rgb)

    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Define the center point of the frame
    center_point = (frame_width // 2, frame_height // 2)

    # Initialize command as "hover"
    command = "hover"

    # Draw landmarks and calculate motion based on the center point
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine gesture and set UAV command
            command = determine_gesture(hand_landmarks, center_point, frame_width, frame_height)
            cv2.putText(frame, f"Gesture: {command.replace('_', ' ').title()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Send the determined UAV command
    send_uav_command(command)

    # Show the current frame
    cv2.imshow("Hand Motion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
