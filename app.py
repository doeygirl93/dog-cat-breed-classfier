import gradio as gr
import torch
import torchvision
from torchvision import transforms

import  model_nn_arch
testset = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=None)
CLASS_NAMES = testset.classes
IMG_SIZE = 224

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model_nn_arch.define_nn_arch().to(DEVICE)
model.load_state_dict(torch.load('models/cat_dog_breed_classfier.pth', map_location=DEVICE))
model.eval()

def predict(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        prediction = model(img)
        prediction_proba = torch.softmax(prediction, dim=1)[0]

    #returns dict of classname n prob
    return {CLASS_NAMES[i]: float(prediction_proba[i]) for i in range (len(CLASS_NAMES))}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Dogo-Cat Breed Classfication toool",
    description="Put in an image and it will classfiy the animals breed!"
)


if __name__ == "__main__":
    interface.launch(share=True)