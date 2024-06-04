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