import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import os
import requests
import time
import serial
from threading import Thread
import base64
import pickle
import gradio as gr
from gradio_client import Client, handle_file

import sounddevice as sd
import soundfile as sf

GOOGLE_API_KEY = "********" 

ELEVENLABS_API_KEY =  "*******"
VOICE_ID = "*******"  

SAVE_DIR = "."
os.makedirs(SAVE_DIR, exist_ok=True)

def speech_to_text(audio_file):
    if not os.path.exists(audio_file):
        print("STT Error: Audio file not found.")
        return "Error: No audio file found."
    
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_API_KEY}"
    payload = {
        "config": {
            "encoding": "MP3",
            "sampleRateHertz": 16000,
            "languageCode": "en-US",
            "alternativeLanguageCodes": ["hi-IN"]
        },
        "audio": {"content": audio_base64}
    }

    response = requests.post(url, json=payload)
    result = response.json()

    try:
        return result["results"][0]["alternatives"][0]["transcript"]
    except:
        print("STT Error:", result)
        return "Error: Could not transcribe audio."

def text_to_speech(response_text):
    if not response_text or response_text.strip() == "":
        print("TTS Error: No response text provided.")
        return None
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    data = {
        "text": response_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.85,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        file_path = os.path.join(SAVE_DIR, "output_audio.mp3")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path  
    else:
        print("TTS Error:", response.text)
        return None


# === Arduino Setup ===
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Change COM port if needed
time.sleep(2)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       model_complexity=1,
                       min_detection_confidence=0.1,
                       min_tracking_confidence=0.1)

# === Global Variables ===
warped_shape = (500, 700)
depth_map = None
depth_map_image = None
colour_map_image = None
perspective_matrix = None
inverse_matrix = None
painting_detected = False
latest_frame = None  # For storing the latest frame globally

# === Helper Functions ===
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_rectangle_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 1000
    for c in contours:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                biggest = approx
                max_area = area
    return biggest

def generate_depth_map_sculptok(image_path, output_path):
    API_KEY = "***********"
    upload_url = "https://api.sculptok.com/api-open/image/upload"
    with open(image_path, "rb") as f:
        files = {"file": f}
        headers = {"apikey": API_KEY}
        upload_response = requests.post(upload_url, files=files, headers=headers)

    if upload_response.status_code != 200 or upload_response.json()["code"] != 0:
        raise Exception("Upload failed: " + upload_response.text)

    image_url = upload_response.json()["data"]["src"]
    submit_url = "https://api.sculptok.com/api-open/draw/prompt"
    payload = {"imageUrl": image_url, "style": "normal"}
    headers.update({"Content-Type": "application/json"})
    submit_response = requests.post(submit_url, json=payload, headers=headers)

    if submit_response.status_code != 200 or submit_response.json()["code"] != 0:
        raise Exception("Submit failed: " + submit_response.text)

    prompt_id = submit_response.json()["data"]["promptId"]
    status_url = f"https://api.sculptok.com/api-open/draw/prompt?uuid={prompt_id}"
    while True:
        status_response = requests.get(status_url, headers=headers)
        data = status_response.json().get("data", {})
        if "imgRecords" in data and len(data["imgRecords"]) >= 1:
            depth_map_url = data["imgRecords"][0]
            break
        time.sleep(5)

    depth_map_image_resp = requests.get(depth_map_url)
    with open(output_path, "wb") as f:
        f.write(depth_map_image_resp.content)

def get_vibration_intensity(x, y, depth_map):
    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
        depth_value = depth_map[y, x] / 255.0
        if depth_value <= 0.01:
            return 0
        return (depth_value ** 1.5) * (1.0 - (1.0 - depth_value) ** 1.5)
    return 0

with open("color_classifier.pkl", "rb") as f: #####ML MODEL
    knn = pickle.load(f)

def get_classified_color(x, y):
    bgr = colour_map_image[y, x]
    rgb = bgr[::-1]  # Convert BGR to RGB
    return knn.predict([rgb])[0]

color_sounds = {
    "Red": "/home/singhasaur/code/HCAI/Project_w_sound/FINAL/sounds1/red_guitar.wav",
    "Blue": "/home/singhasaur/code/HCAI/Project_w_sound/FINAL/sounds1/blue_flute.wav",
    "Green": "/home/singhasaur/code/HCAI/Project_w_sound/FINAL/sounds1/green_conga.wav",
    "Yellow": "/home/singhasaur/code/HCAI/Project_w_sound/FINAL/sounds1/yellow_piano.wav",
    "Black": "/home/singhasaur/code/HCAI/Project_w_sound/FINAL/sounds1/black_sax.wav",  
}

last_color = None

def play_sound(color):
    global last_color
    if color == "White":
        sd.stop()
        last_color = "White"
        return 
    if color in color_sounds:
        sound_file = color_sounds[color]
    if color == last_color:
        return
    last_color = color  # Update last color
    data, samplerate = sf.read(sound_file)
    sd.play(data, samplerate, blocking=False)  

def stream_live():
    global depth_map, depth_map_image, perspective_matrix, inverse_matrix
    global colour_map_image, perspective_matrix_colour, color_overlay
    global painting_detected, latest_frame

    cap = cv2.VideoCapture('/dev/video2')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        latest_frame = frame.copy()

        # Frame 1: Raw display
        frame_normal = frame.copy()
        display_frame = frame.copy()

        if painting_detected:
            overlay = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Overlay colour image (projected)
            if inverse_matrix is not None and colour_map_image is not None:
                projected_colour = cv2.warpPerspective(colour_map_image, inverse_matrix, (frame.shape[1], frame.shape[0]))
                mask_colour = cv2.warpPerspective(np.ones(colour_map_image.shape[:2], dtype=np.uint8) * 255,
                                                  inverse_matrix, (frame.shape[1], frame.shape[0]))
                mask_3c_colour = cv2.merge([mask_colour, mask_colour, mask_colour])
                inv_mask_colour = cv2.bitwise_not(mask_3c_colour)
                background_colour = cv2.bitwise_and(frame_normal, inv_mask_colour)
                foreground_colour = cv2.bitwise_and(projected_colour, mask_3c_colour)
                frame_normal = cv2.add(background_colour, foreground_colour)

            # Overlay depth image (projected)
            if inverse_matrix is not None and depth_map_image is not None:
                projected = cv2.warpPerspective(depth_map_image, inverse_matrix, (frame.shape[1], frame.shape[0]))
                mask = cv2.warpPerspective(np.ones(depth_map_image.shape[:2], dtype=np.uint8) * 255,
                                           inverse_matrix, (frame.shape[1], frame.shape[0]))
                mask_3c = cv2.merge([mask, mask, mask])
                inv_mask = cv2.bitwise_not(mask_3c)
                background = cv2.bitwise_and(overlay, inv_mask)
                foreground = cv2.bitwise_and(projected, mask_3c)
                overlay = cv2.add(background, foreground)

            # Hand tracking and pointer
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    cx = int(handLms.landmark[8].x * w)
                    cy = int(handLms.landmark[8].y * h)

                    # ✅ Accurate color detection on warped colour overlay
                    if perspective_matrix_colour is not None and color_overlay is not None:
                        src_pt = np.array([[[cx, cy]]], dtype='float32')
                        warped_pt = cv2.perspectiveTransform(src_pt, perspective_matrix_colour)
                        x_warped, y_warped = warped_pt[0][0].astype(int)

                        x_clamped = np.clip(x_warped, 0, color_overlay.shape[1] - 1)
                        y_clamped = np.clip(y_warped, 0, color_overlay.shape[0] - 1)

                        color_name = get_classified_color(x_clamped, y_clamped)
                        play_sound(color_name)

                    # Vibration feedback from depth map
                    if perspective_matrix is not None:
                        src_pt = np.array([[[cx, cy]]], dtype='float32')
                        warped_pt = cv2.perspectiveTransform(src_pt, perspective_matrix)
                        x_w, y_w = warped_pt[0][0].astype(int)
                        intensity = get_vibration_intensity(x_w, y_w, depth_map)
                        pwm_value = int(np.clip(intensity * 255, 0, 255))
                        arduino.write(f"{pwm_value}\n".encode())

                    # Pointer on both overlays
                    cv2.circle(overlay, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.circle(frame_normal, (cx, cy), 10, (0, 255, 0), -1)

            display_frame = overlay
        else:
            cnt = get_rectangle_contour(display_frame)
            cnt = get_rectangle_contour(frame_normal)
            if cnt is not None:
                cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 3)
                cv2.drawContours(frame_normal, [cnt], -1, (0, 255, 0), 3)

        # Convert to RGB for display
        frame_normal_rgb = cv2.cvtColor(frame_normal, cv2.COLOR_BGR2RGB)
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        yield frame_normal_rgb, display_frame_rgb  # 🔥 Yield both frames

    cap.release()

# === Called when Detect Painting button is clicked ===
def detect_painting():
    global painting_detected, depth_map, depth_map_image, perspective_matrix
    global inverse_matrix, latest_frame, colour_map_image
    global perspective_matrix_colour, color_overlay  # 👈 Add new globals

    if painting_detected:
        return "✅ Already detected. Refresh or reset to try again."

    if latest_frame is None:
        return "⚠️ No frame available yet. Please wait."

    cnt = get_rectangle_contour(latest_frame)
    if cnt is None:
        return "❌ No rectangular painting found. Try again."

    # Warp the detected painting
    pts = cnt.reshape(4, 2)
    ordered = order_points(pts)
    dst_pts = np.array([[0, 0], [warped_shape[0] - 1, 0],
                        [warped_shape[0] - 1, warped_shape[1] - 1],
                        [0, warped_shape[1] - 1]], dtype="float32")

    # Perspective transforms
    perspective_matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
    inverse_matrix = cv2.getPerspectiveTransform(dst_pts, ordered)

    warped = cv2.warpPerspective(latest_frame, perspective_matrix, warped_shape)
    cv2.imwrite("warped_reference.png", warped)

    # 🔍 Load or Generate Depth Map
    if os.path.exists("test_depth.png"):
        print("[INFO] Loading existing depth map...")
    else:
        print("[INFO] Generating new depth map via SculptOK...")
        generate_depth_map_sculptok("warped_reference.png", "test_depth.png")

    depth_map_image = cv2.imread("test_depth.png")
    depth_map_image = cv2.resize(depth_map_image, warped_shape)
    depth_map = cv2.cvtColor(depth_map_image, cv2.COLOR_BGR2GRAY)

    # 🎨 Load and warp colour map
    colour_map_image = cv2.imread("test_colour.png")
    colour_map_image = cv2.resize(colour_map_image, warped_shape)

    perspective_matrix_colour = cv2.getPerspectiveTransform(ordered, dst_pts)
    color_overlay = cv2.warpPerspective(colour_map_image, perspective_matrix_colour, warped_shape)

    # ✅ Detection complete
    painting_detected = True
    return "✅ Painting detected. Depth and colour maps ready!"



# Assuming speech_to_text and text_to_speech are already defined in your notebook
chat_history = []

def get_chatbot_response(question, image_path):
    client = Client(GRADIO_API_URL)
    response = client.predict(
        handle_file(image_path),
        question
    )
    return response

def voice_chatbot(audio_file):
    global chat_history

    # Convert speech to text
    question = speech_to_text(audio_file)

    if question.startswith("Error"):
        chat_history.append(("User (voice)", question))
        return chat_history, "", None

    image_path = "warped_reference.png"

    try:
        answer = get_chatbot_response(question, image_path)
    except Exception as e:
        answer = f"Error contacting API: {str(e)}"

    chat_history.append((question, answer))

    # Convert chatbot answer to speech
    audio_response_path = text_to_speech(answer)

    return chat_history, answer, audio_response_path

def text_chatbot(user_text):
    global chat_history

    image_path = "warped_reference.png"
    question = user_text

    try:
        answer = get_chatbot_response(question, image_path)
    except Exception as e:
        answer = f"Error contacting API: {str(e)}"

    chat_history.append((question, answer))

    audio_response_path = text_to_speech(answer)

    return chat_history, answer, audio_response_path

GRADIO_API_URL = "https://00e4dffac7d946c1bd.gradio.live"
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align:center">
            <h1>Odin Vision</h1>
            <h3>OdinVision is an assistive technology system designed to help visually impaired individuals perceive and interpret visual artworks—particularly paintings—through a multisensory experience powered by real-time computer vision, depth mapping, and human-computer interaction techniques.At its core, OdinVision captures a live video feed of the environment and uses advanced depth sensing to generate a real-time depth map of the scene. This allows the system to accurately estimate the spatial layout and proximity of objects, particularly flat surfaces like paintings. Once a painting is detected and the depth data is obtained, the system overlays this spatial information with color recognition and hand-tracking to create an interactive experience.</h3>
        </div>
        
        """
    )

    with gr.Row():
        image_output1 = gr.Image(label="Colour Overlay View", elem_id="image_output1")
        image_output2 = gr.Image(label="Depth Overlay View", elem_id="image_output2")

    detect_btn = gr.Button("Detect Painting")
    status_box = gr.Textbox(label="Status")

    demo.load(fn=stream_live, inputs=[], outputs=[image_output1, image_output2])
    detect_btn.click(fn=detect_painting, inputs=[], outputs=status_box)

    #chatbot---------------------------------------------------------------------------------
    chatbot = gr.Chatbot(label="Painting Chatbot")

    # --- Voice Input Section ---
    gr.Markdown(
        """<h2>Ask by Voice</h2>
        <p> Use the button below to convey your questions to the chatbox verbally</p>"""
    )
    audio_input = gr.Audio(type="filepath", label="Voice Question")
    send_voice_btn = gr.Button("Send Voice Question")

    # --- Text Input Section ---
    gr.Markdown(
        """<h2>Ask by Text</h2>
        <p> Use the box below to convey your questions to the chatbox textually</p>"""
    )
    text_input = gr.Textbox(label="Text Question", placeholder="Type your question here...")
    send_text_btn = gr.Button("Send Text Question")

    # Outputs
    response_textbox = gr.Textbox(label="Textual Response", visible=False)
    audio_output = gr.Audio(label="Voice Response")

    # # Webcam streaming and painting detection
    # demo.load(fn=stream_live, inputs=[], outputs=image_output)  # Updated to use gr.Image
    # detect_btn.click(fn=detect_painting, inputs=[], outputs=status_box)

    # Voice chatbot interaction
    send_voice_btn.click(
        fn=voice_chatbot,
        inputs=audio_input,
        outputs=[chatbot, response_textbox, audio_output]
    )

    # Text chatbot interaction
    send_text_btn.click(
        fn=text_chatbot,
        inputs=text_input,
        outputs=[chatbot, response_textbox, audio_output]
    )

    #add a feeback form field and submit button that gets saved to a csv file
    feedback_input = gr.Textbox(label="Feedback", placeholder="Type your feedback here...")
    submit_feedback_btn = gr.Button("Submit Feedback")
    feedback_output = gr.Textbox(label="Feedback Status")
    submit_feedback_btn.click(
        fn=lambda feedback: save_feedback(feedback),
        inputs=feedback_input,
        outputs=feedback_output
    )
    # Save feedback to CSV
    def save_feedback(feedback):
        with open("feedback.csv", "a") as f:
            f.write(f"{feedback}\n")
        return "Feedback submitted successfully!"
    


demo.launch()
