import numpy as np
import random
import csv

def rotate_landmarks(landmarks, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T)
    return rotated_landmarks

def main():
    with open('hand.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  

        with open('synthetic.csv', 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['class', 'landmarks'])

            for row in reader:
                class_label = int(row[0])
                landmarks_str = row[1].strip('][').split(', ')
                landmarks = np.array([float(x) for x in landmarks_str]).reshape(-1, 2)

                for i in range(15):
                    random_angle = random.uniform(-180, 180)
                    rotated_landmarks = rotate_landmarks(landmarks, random_angle)
                    rotated_landmarks_vector = rotated_landmarks.flatten().tolist()


                    writer.writerow([class_label, rotated_landmarks_vector])

if __name__ == "__main__":
    main()
