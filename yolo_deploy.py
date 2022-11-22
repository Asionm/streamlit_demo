from PIL import Image
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
import torch
import time
import streamlit as st
import cv2
import tempfile
import numpy as np



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
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE)

model_equation = torch.hub.load('./yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True)

def get_code_text(path):
    with open(path,encoding='utf8') as f:
        return f.read()


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

def get_prediction(img, threshold):
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    img = img.to(DEVICE)
    # model的返回结果
    pred = model([img])  # pred包含了预测的边框顶点、类型和置信度
    # 预测的类型
    pred_class = [inst_classes[i] for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    # 方框的位置
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    # 置信度(注意此处分数已经按从高到低排列)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class
    except IndexError:
        return 0, 0

def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    boxes, pred_cls = get_prediction(img, threshold)  # Get predictions
    if boxes == 0:
        return 0
    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        boxes[i][0] = tuple(map(lambda x:int(x),boxes[i][0]))
        boxes[i][1] = tuple(map(lambda x: int(x), boxes[i][1]))
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        # Write the prediction class
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return img

def start_video(path):
    # cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(path)  # 打开视频流(若path=0表示开启摄像头流)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的搞
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码

    # 视频对象的输出
    out = cv2.VideoWriter('./123.mp4', fourcc, 20.0, (width, height))

    while cap.isOpened():
        # 读取一帧数据，一帧就是一张图
        ok, frame = cap.read()
        if not ok:
            break
        frame = object_detection_api(frame, 0.8)
        try:
            if len(frame) == 1:
                print(frame.all())
                continue
        except (AttributeError, TypeError):
            if frame == 0:
                continue
        # 输入'q'退出程序
        # cv2.imshow(window_name, frame)
        out.write(frame)
        c = cv2.waitKey(1)  # 延时1ms切换到下一帧图像
        if c & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    # out.release()
    cv2.destroyAllWindows()





def plot_boxes(results, frame):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, "equation", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
    return frame

def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

def pic_run():
    st.title("图像物体识别")
    st.write("")
    file_up = st.file_uploader("请上传图片", type = "jpg")
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = '已上传的图片', use_column_width = True)
        st.write("")
        predict(model,file_up)

def video_run():
    st.title("视频物体识别")
    place_holder = st.empty()
    file_up = place_holder.file_uploader("请上传视频", type="mp4")
    if file_up is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_up.read())

        place_holder.empty()
        place_holder.video(file_up, format="video/mp4", start_time=0)
        start_bnt = st.button("上传完成点击开始识别")
        if start_bnt:
            place_holder.write("由于计算资源有限请耐心等待....")
            start_video(tfile.name)
            place_holder.empty()
            st.write("计算完成点击查看")
            place_holder.video('123.mp4')

def equation_run():
    st.title("算式识别")
    st.write("")
    file_up = st.file_uploader("请上传图片", type="jpg")
    place_holder = st.empty()
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        place_holder.image(image, caption='已上传的图片', use_column_width=True)
        place_holder.write("")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model_equation)  ### DETECTION HAPPENING HERE
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame)
        place_holder.empty()
        place_holder.image(frame)

st.sidebar.title("功能选择")

app_mode = st.sidebar.selectbox("模式选择",
    ["图像物体识别", "视频物体识别", "展示源代码","算式识别"])
if app_mode == "图像物体识别":
    pic_run()
elif app_mode == "展示源代码":
    st.code(get_code_text("yolo_deploy.py"))
elif app_mode == "视频物体识别":
    video_run()
elif app_mode == "算式识别":
    equation_run()


