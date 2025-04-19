from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pyautogui

app = Flask(__name__)

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)
current_mode = "hand"  # Default mode
tracking_active = False

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_hand(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Get the centroid of the contour
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw the contour and centroid
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            if tracking_active:
                # Map hand position to screen coordinates
                screen_x = np.interp(cx, [0, frame.shape[1]], [0, screen_width])
                screen_y = np.interp(cy, [0, frame.shape[0]], [0, screen_height])
                pyautogui.moveTo(screen_x, screen_y)
    
    return frame

def detect_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2 and tracking_active:
            # Calculate center point between eyes
            eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                cv2.circle(frame, eye_center, 2, (0, 255, 0), -1)
                eye_centers.append(eye_center)
            
            # Use average position of eyes for cursor control
            avg_x = sum(c[0] for c in eye_centers) / len(eye_centers)
            avg_y = sum(c[1] for c in eye_centers) / len(eye_centers)
            
            # Map eye position to screen coordinates
            screen_x = np.interp(avg_x, [0, frame.shape[1]], [0, screen_width])
            screen_y = np.interp(avg_y, [0, frame.shape[0]], [0, screen_height])
            pyautogui.moveTo(screen_x, screen_y)
    
    return frame

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        
        if current_mode == "hand":
            frame = detect_hand(frame)
        else:
            frame = detect_face_and_eyes(frame)
        
        # Add mode indicator
        cv2.putText(frame, f"Mode: {current_mode}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {'Active' if tracking_active else 'Inactive'}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 255, 0) if tracking_active else (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.get_json()
    current_mode = data['mode']
    return jsonify({'status': 'success'})

@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    global tracking_active
    tracking_active = not tracking_active
    return jsonify({'status': 'success', 'tracking': tracking_active})

if __name__ == "__main__":
    app.run(debug=True) 