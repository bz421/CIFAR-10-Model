import tkinter as tk
from tkinter import filedialog, messagebox

import filetype
import torch
from PIL import Image, ImageTk
from torchvision import datasets, transforms
from tqdm import tqdm

from resnet import ResNet18


def getDataLoaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def loadModel(modelPath):
    model = ResNet18()
    state_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    model.eval()
    return model

def evaluateModel(model, max_examples=100):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    misclassified_images = []
    misclassified_predictions = []
    misclassified_labels = []
    misclassified_confidences = []

    all_predictions = []
    all_targets = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Evaluating'):
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]

            for idx in misclassified_indices:
                if len(misclassified_images) < max_examples:
                    misclassified_images.append(images[idx].cpu())
                    misclassified_predictions.append(predicted[idx].item())
                    misclassified_labels.append(labels[idx].item())

                    softmax_probs = torch.nn.functional.softmax(outputs[idx], dim=0)
                    confidence = softmax_probs[predicted[idx]].item()
                    misclassified_confidences.append(confidence)

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    return {
        'accuracy': accuracy,
        'classes': classes
    }

class App:
    def __init__(self, modelPath):
        self.model = loadModel(modelPath)

        # result = evaluateModel(self.model)
        # accuracy, classes = result['accuracy'], result['classes']
        # print(accuracy, classes)

        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.root = tk.Tk()
        self.root.title('Image Classifier')
        self.root.geometry('600x500')

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        self.pred_label = tk.Label(
            self.root,
            text='Upload an image to classify',
            font=('Helvetica', 18)
        )
        self.pred_label.pack(pady=10)

        self.upload_button = tk.Button(self.root, text='Upload Image', command=self.upload_image)
        self.upload_button.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ('Image Files', '*.png *.jpg *.jpeg *.gif *.bmp *.webp')
            ]
        )

        print(filetype.guess(file_path), filetype.guess(file_path).extension)
        if not file_path: return

        try:
            original_image = Image.open(file_path)
            # save to orig.png
            original_image.save(f'images/orig.{filetype.guess(file_path).extension}')

            display_image = original_image.copy()
            display_image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            input_image = original_image.convert('RGB')
            input_tensor = self.transform(input_image).unsqueeze(0)

            processed_image = transforms.ToPILImage()(input_tensor.squeeze(0))
            processed_image.save('images/processed.png')

            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                prediction = self.classes[predicted.item()]

            self.pred_label.config(text=f'Prediction: {prediction}')

        except Exception as e:
            messagebox.showerror('Error', f'Error: {e}')

if __name__ == '__main__':
    app = App('model5-distilled/resnet18_distill_best.pt')
    app.root.mainloop()