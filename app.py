from flask import Flask, render_template, request
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
from werkzeug.utils import secure_filename


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 512, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size()[0], 512 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


device = torch.device('cpu')

model = torch.load("./model/Hoo_Hacks_model.pth", map_location=device)
model.eval()


def model_predict(img_path):
    img = Image.open(img_path)
    test_transform = transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = test_transform(img)
    image = image.to(device)
    return torch.argmax(model(image.unsqueeze(0)))


app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/team')
def team():
    return render_template('index.html')


@app.route('/statistics')
def statistics():
    return render_template('statistics.html')


@app.route('/viz')
def tableau():
    return render_template('tableau.html')


@app.route('/model')
def classification():
    return render_template('information.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        if preds == 0:
            return "Compost Material"
        return "Recyclable Material"
    return None


if __name__ == '__main__':
    app.run(debug=True)
