import os
import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
video_folder = "videos"
output_folder = "hands"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

target_size = (300, 300)

for video_subfolder in os.listdir(video_folder):
    contatoreVideo=0
    for video_file in os.listdir(os.path.join(video_folder, video_subfolder)):
        contatoreVideo+=1
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            video_path = os.path.join(video_folder, video_subfolder, video_file)
            output_subfolder = os.path.join(output_folder, video_subfolder)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                # Legge il frame corrente
                ret, frame = cap.read()
                if not ret:
                    break

                # Incrementa il conteggio dei frame
                frame_count += 1

                # Converti il frame in RGB per MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Esegue il rilevamento delle mani su MediaPipe
                results = hands.process(frame_rgb)

                # Esegue il cropping delle mani e le salva come immagini
                if results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Estrai le coordinate delle landmark delle mani
                        hand_points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]

                        # Calcola il rettangolo delimitatore delle mani
                        x, y, w, h = cv2.boundingRect(np.array(hand_points))

                        # Calcola il centro del rettangolo
                        center_x = x + w // 2
                        center_y = y + h // 2

                        # Calcola i nuovi punti per ottenere un ritaglio 300x300
                        new_x = max(0, center_x - target_size[0] // 2)
                        new_y = max(0, center_y - target_size[1] // 2)
                        new_x_end = min(frame.shape[1], new_x + target_size[0])
                        new_y_end = min(frame.shape[0], new_y + target_size[1])

                        # Esegue il cropping della mano e la salva come immagine
                        hand_crop = frame[new_y:new_y_end, new_x:new_x_end]

                        # Ridimensiona il ritaglio alla dimensione desiderata
                        hand_crop_resized = cv2.resize(hand_crop, target_size)

                        cv2.imwrite(os.path.join(output_subfolder, f"{video_subfolder}_frame_{frame_count}_hand_{i}_video{contatoreVideo}.png"), hand_crop_resized)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Chiude il video
            cap.release()

# Chiude tutte le finestre
cv2.destroyAllWindows()
