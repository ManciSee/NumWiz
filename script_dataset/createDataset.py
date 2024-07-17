import os
import cv2
import mediapipe as mp
import csv
import numpy as np

hand_directory = 'hand'
output_directory = "hand_cropped"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data = []
labels = []

def crop_and_resize_hand(image, hand_landmarks, margin=20, size=400):
    x_min, y_min = min(hand_landmarks, key=lambda x: x[0])[0], min(hand_landmarks, key=lambda x: x[1])[1]
    x_max, y_max = max(hand_landmarks, key=lambda x: x[0])[0], max(hand_landmarks, key=lambda x: x[1])[1]
    
    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    
    cropped_hand = image[y_min:y_max, x_min:x_max]
    
    # Resize the cropped hand to fit within the size while maintaining aspect ratio
    height, width, _ = cropped_hand.shape
    if height > width:
        new_height = size
        new_width = int(size * width / height)
    else:
        new_width = size
        new_height = int(size * height / width)
    
    resized_cropped_hand = cv2.resize(cropped_hand, (new_width, new_height))
    
    # Create a black background image
    background = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Calculate the position to place the resized cropped hand on the background
    x_start = (size - new_width) // 2
    y_start = (size - new_height) // 2
    x_end = x_start + new_width
    y_end = y_start + new_height
    
    # Place the resized cropped hand on the background
    background[y_start:y_end, x_start:x_end] = resized_cropped_hand
    
    return background

img_counter = 0
for img_file in os.listdir(hand_directory):
    img_path = os.path.join(hand_directory, img_file)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Error loading image: {img_path}")
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in hand_landmarks.landmark]
            cropped_hand = crop_and_resize_hand(image, hand_landmarks_points)
            
            if cropped_hand.size > 0:
                output_filename = f"cropped_{img_counter}.jpg"
                cv2.imwrite(os.path.join(output_directory, output_filename), cropped_hand)
                print(f"Hand cropped from {img_path} and saved to {os.path.join(output_directory, output_filename)}")
                
                data_aux = []
                for point in hand_landmarks.landmark:
                    data_aux.extend([point.x, point.y])
                data.append(data_aux)
                
                img_counter += 1
            else:
                print(f"No hand detected in {img_path}")
    else:
        print(f"No hand detected in {img_path}")

hands.close()


# print("Saving data to CSV file...")
# with open("hand.csv", "w", newline='') as csvfile:
#     fieldnames = ['class', 'landmarks']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for label, landmarks in zip(labels, data):
#         writer.writerow({'class': label, 'landmarks': landmarks})

print("Done!")


# This is the script to collect data from the webcam
"""
import os
import cv2
import mediapipe as mp
import csv

# hand_directory = 'hand_dataset'
# if not os.path.exists(hand_directory):
#     os.makedirs(hand_directory)

number_of_classes = 4
# dataset_size = 100

# cap = cv2.VideoCapture(1)

# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(hand_directory, str(j))):
#         os.makedirs(os.path.join(hand_directory, str(j)))

#     print('Collecting data for class {}'.format(j))

#     counter = 0
#     while counter < dataset_size:
#         input('Press enter to start collecting data')
#         while True:
#             ret, frame = cap.read()
#             if frame is None:
#                 print("Frame not captured. Check your camera.")
#                 break
#             cv2.imshow('frame', frame)
#             if cv2.waitKey(25) == ord('q'):
#                 break
#             cv2.imwrite(os.path.join(hand_directory, str(j), '{}.jpg'.format(counter)), frame)
#             counter += 1
#             if counter == dataset_size:
#                 break
#         frame = cv2.putText(frame, 'Press enter to continue', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         cv2.waitKey(2000)  

# cap.release()
# cv2.destroyAllWindows()

# print('Data collection completed')

# -------- Crop data --------
print()
print("Starting crop data...")

hand_directory = 'hand_dataset'
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data = []
labels = []

def crop_hand(image, hand_landmarks, size=400):
    x_min, y_min = min(hand_landmarks, key=lambda x: x[0])[0], min(hand_landmarks, key=lambda x: x[1])[1]
    x_max, y_max = max(hand_landmarks, key=lambda x: x[0])[0], max(hand_landmarks, key=lambda x: x[1])[1]
    x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
    x_min = max(0, x_center - size // 2)
    y_min = max(0, y_center - size // 2)
    x_max = min(image.shape[1], x_center + size // 2)
    y_max = min(image.shape[0], y_center + size // 2)
    cropped_hand = image[y_min:y_max, x_min:x_max]
    return cropped_hand

output_directory = "hand_cropped"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for subdir in range(number_of_classes):
    subdir_path = os.path.join(hand_directory, str(subdir))

    for img_num in range(100):
        img_path = os.path.join(subdir_path, f"{img_num}.jpg")
        data_aux = []
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(subdir)

            hand_landmarks = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in results.multi_hand_landmarks[0].landmark]
            cropped_hand = crop_hand(image, hand_landmarks)
        
            if cropped_hand.size > 0:
                subdir_output = os.path.join(output_directory, str(subdir))
                if not os.path.exists(subdir_output):
                    os.makedirs(subdir_output)
                
                resized_cropped_hand = cv2.resize(cropped_hand, (400, 400))
                cv2.imwrite(os.path.join(subdir_output, f"cropped_hand_{subdir}_{img_num}.jpg"), resized_cropped_hand)
                print(f"Hand cropped from {img_path} and saved to {subdir_output}")
            else:
                print(f"No hand detected in {img_path}")
        else:
            print(f"No hand detected in {img_path}")

hands.close()
print()

print("Saving data to CSV file...")
with open("hand.csv", "w", newline='') as csvfile:
    fieldnames = ['class', 'landmarks']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for l, d in zip(labels, data):
        writer.writerow({'class': l, 'landmarks': d})

print("Done!")
"""
