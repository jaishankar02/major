import gradio as gr
import torch
import cv2
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

def predict(image):
    img = cv2.resize(image, (256,256))
    img = torch.tensor(img).permute(2,0,1).float()/255.0
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    return pred

def show_plots():
    return ["outputs/loss.png", "outputs/miou.png", "outputs/mdice.png"]

iface1 = gr.Interface(fn=show_plots,
                      inputs=[],
                      outputs=[gr.Image(), gr.Image(), gr.Image()],
                      title="Training Metrics")

iface2 = gr.Interface(fn=predict,
                      inputs=gr.Image(),
                      outputs=gr.Image(),
                      title="Segmentation")

app = gr.TabbedInterface([iface1, iface2],
                        ["Metrics", "Prediction"])

app.launch()