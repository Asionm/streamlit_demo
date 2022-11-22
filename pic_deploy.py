from PIL import Image
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
import torch
import time
import streamlit as st

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,progress=True)

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

def data_preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()])
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    imgg = transform(img)
    return batch_t, imgg

def predict(model,image):
    batch_t, img = data_preprocess(image)
    time_start = time.time()
    model.eval()
    outputs = model(batch_t)
    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Just', time_sum, 'second!')
    st.write(outputs)

    time_start = time.time()
    # draw bboxes,labels on the raw input image for the object candidates with score larger than score_threshold
    score_threshold = .8
    st.write([inst_classes[label] for label in outputs[0]['labels'][outputs[0]['scores'] > score_threshold]])
    output_labels = [inst_classes[label] for label in outputs[0]['labels'][outputs[0]['scores'] > score_threshold]]
    output_boxes = outputs[0]['boxes'][outputs[0]['scores'] > score_threshold]
    images = img * 255.0;
    images = images.byte()
    result = draw_bounding_boxes(images, boxes=output_boxes, labels=output_labels, width=5)
    st.image(result.permute(1, 2, 0).numpy(), caption='Processed Image.', use_column_width=True)
    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Draw', time_sum, 'second!')
    return outputs

st.title("Simple Object Detection Application")
st.write("")
file_up = st.file_uploader("Upload an image", type = "jpg")
if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    labels = predict(model,file_up)