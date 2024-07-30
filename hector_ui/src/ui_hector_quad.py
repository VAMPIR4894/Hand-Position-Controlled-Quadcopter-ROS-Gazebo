#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import rospy
from geometry_msgs.msg import Twist

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize ROS node and publisher
rospy.init_node('uav_control_node')
twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

def init_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera with index {index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def send_uav_command(command):
    twist_msg = Twist()
    if command == "hover":
        pass
    elif command == "go_up":
        twist_msg.linear.z = 1
    elif command == "go_down":
        twist_msg.linear.z = -1
    elif command == "go_forward":
        twist_msg.linear.x = 1
    elif command == "go_backward":
        twist_msg.linear.x = -1
    elif command == "go_left":
        twist_msg.linear.y = -1
    elif command == "go_right":
        twist_msg.linear.y = 1
    elif command == "turn_clockwise":
        twist_msg.angular.z = -1
    elif command == "turn_counter_clockwise":
        twist_msg.angular.z = 1

    twist_pub.publish(twist_msg)

def determine_gesture(hand_landmarks, center_point, frame_width, frame_height):
    wrist = hand_landmarks.landmark[0]
    wrist_point = (int(wrist.x * frame_width), int(wrist.y * frame_height))
    
    edge_threshold = 0.1

    left_edge = frame_width * edge_threshold
    right_edge = frame_width * (1 - edge_threshold)
    top_edge = frame_height * edge_threshold
    bottom_edge = frame_height * (1 - edge_threshold)

    if wrist_point[1] < top_edge:
        if wrist_point[0] < left_edge:
            return "turn_counter_clockwise"
        elif wrist_point[0] > right_edge:
            return "turn_clockwise"
        else:
            return "go_up"
    elif wrist_point[1] > bottom_edge:
        if wrist_point[0] < left_edge:
            return "go_left"
        elif wrist_point[0] > right_edge:
            return "go_right"
        else:
            return "go_down"
    elif wrist_point[1] < center_point[1] and wrist_point[1] >= top_edge:
        if wrist_point[0] < left_edge:
            return "turn_counter_clockwise"
        elif wrist_point[0] > right_edge:
            return "turn_clockwise"
        else:
            return "go_forward"
    elif wrist_point[1] > center_point[1] and wrist_point[1] <= bottom_edge:
        if wrist_point[0] < left_edge:
            return "go_left"
        elif wrist_point[0] > right_edge:
            return "go_right"
        else:
            return "go_backward"
    else:
        if wrist_point[0] < left_edge:
            return "go_left"
        elif wrist_point[0] > right_edge:
            return "go_right"
        else:
            return "hover"

cap = None
for i in range(5):
    cap = init_camera(i)
    if cap:
        break

if not cap:
    print("Failed to open camera. Exiting...")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    frame_height, frame_width, _ = frame.shape
    center_point = (frame_width // 2, frame_height // 2)

    command = "hover"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            command = determine_gesture(hand_landmarks, center_point, frame_width, frame_height)
            cv2.putText(frame, f"Gesture: {command.replace('_', ' ').title()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    send_uav_command(command)
    cv2.imshow("Hand Motion Detection", frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

