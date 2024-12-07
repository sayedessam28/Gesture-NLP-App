#!/usr/bin/env python
# coding: utf-8

# In[6]:


import mediapipe as mp
import cv2
import time
from transformers import pipeline
import streamlit as st

# Initialize NLP models
st.write("Loading models, please wait...")
summarizer = pipeline("summarization", model="t5-small")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
st.write("Models loaded successfully!")

# Cooldown mechanism
last_action_time = 0
cooldown_period = 2  # Seconds between actions

# Function to process gestures
def perform_action(gesture, input_text):
    global last_action_time
    current_time = time.time()

    if current_time - last_action_time < cooldown_period:
        print(f"Cooldown active. Gesture '{gesture}' ignored.")
        return

    last_action_time = current_time  # Update the last action time

    if gesture == "Swipe Up":
        st.write("Gesture Detected: **Swipe Up** - Summarizing Text...")
        try:
            output = summarizer(input_text, max_length=50, min_length=25, do_sample=False)
            st.write("**Summarized Text:**", output[0]['summary_text'])
        except Exception as e:
            st.write(f"Error in summarization: {e}")
    elif gesture == "Swipe Down":
        st.write("Gesture Detected: **Swipe Down** - Translating Text...")
        try:
            output = translator(input_text)
            st.write("**Translated Text (Arabic):**", output[0]['translation_text'])
        except Exception as e:
            st.write(f"Error in translation: {e}")

# Function to detect gestures
def detect_gesture(hand_landmarks):
    if hand_landmarks:
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        if index_tip.y < thumb_tip.y:  # Swipe Up
            return "Swipe Up"
        elif index_tip.y > thumb_tip.y:  # Swipe Down
            return "Swipe Down"
    return None

# Streamlit UI
st.title("Gesture-Controlled NLP App")
st.write("Control text summarization and translation with hand gestures!")
input_text = st.text_area("Input Text", placeholder="Enter or paste text here...")
run_app = st.checkbox("Start Gesture Detection")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

if run_app and input_text.strip():
    cap = cv2.VideoCapture(0)
    st.write("Webcam started. Perform gestures in front of the camera.")

    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Gesture detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected_gesture = detect_gesture(hand_landmarks.landmark)
                if detected_gesture:
                    perform_action(detected_gesture, input_text)

        # Display webcam feed
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Enter text and start the app to begin!")


# In[ ]:




