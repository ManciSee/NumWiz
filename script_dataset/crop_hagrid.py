import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from mediapipe import audio

# Define the names of the classes of interest
class_names = ["fist", "three2", "peace", "palm", "four", "one"] # then rename these names to the names of their classes (0,1,2,...)

hand_directory = 'PATH_TO_HAGRID_DATASET'
output_directory = "PATH_TO_OUTPUT_DIRECTORY"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def crop_hand(image, hand_landmarks, margin=20):
    x_min, y_min = min(hand_landmarks, key=lambda x: x[0])[0], min(hand_landmarks, key=lambda x: x[1])[1]
    x_max, y_max = max(hand_landmarks, key=lambda x: x[0])[0], max(hand_landmarks, key=lambda x: x[1])[1]
    
    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    
    cropped_hand = image[y_min:y_max, x_min:x_max]
    
    return cropped_hand

for class_name in tqdm(class_names, desc="Processing classes"):
    class_path = os.path.join(hand_directory, class_name)
    if not os.path.exists(class_path):
        continue  # Salta se la directory della classe non esiste
    
    img_counter = 0
    img_files = os.listdir(class_path)
    img_files = img_files[:4000]  # Limit the number of image for each classes

    with tqdm(total=len(img_files), desc=f"Processing {class_name}", leave=False) as pbar:
        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Error loading image: {img_path}")
                pbar.update(1)
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_landmarks_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in hand_landmarks.landmark]
                    cropped_hand = crop_hand(image, hand_landmarks_points)
                    
                    if cropped_hand.size > 0:
                        class_output_dir = os.path.join(output_directory, class_name)
                        if not os.path.exists(class_output_dir):
                            os.makedirs(class_output_dir)
                        
                        output_filename = f"{class_name}_{img_counter}.jpg"
                        cv2.imwrite(os.path.join(class_output_dir, output_filename), cropped_hand)
                        
                        img_counter += 1
                    else:
                        print(f"No hand detected in {img_path}")
            else:
                print(f"No hand detected in {img_path}")

            pbar.update(1)

hands.close()
