import torch
from torchvision import transforms, models
from PIL import Image
import os
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)  # Assuming 6 classes as per the training setup
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('hand_gesture_cnn_final_res18_finale.pth', map_location=device))
model.to(device)
model.eval()
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  
    image = image.to(device)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

images_dir = 'images'

predictions = {}
for img_file in os.listdir(images_dir):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(images_dir, img_file)
        prediction = predict_image(img_path, model, transform, device)
        predictions[img_file] = prediction

for img_file, pred in predictions.items():
    print(f'Image: {img_file}, Predicted class: {pred}')
