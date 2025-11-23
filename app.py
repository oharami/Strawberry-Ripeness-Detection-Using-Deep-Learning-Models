import os
import json
import cv2
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn

import os
import sys
import shutil
import threading
import warnings

import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
CORS(app)  # 解决跨域问题

app.jinja_env.variable_start_string = '<<'
app.jinja_env.variable_end_string = '>>'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_size = 450
img2predict = ""
model_path = "runs/train/exp/weights/best.pt"


@torch.no_grad()
def model_load(weights="",  # model.pt path(s)
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               ):
    """模型初始化"""
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    print("Model loading completed!")
    return model


model = model_load(weights=model_path, device=device)


@app.route('/upload', methods=['get', 'post'])
def upload_picture():
    file_name = request.files['file'].filename
    target_image_name = 'images/tmp/tmp_upload.' + file_name.split(".")[-1]
    request.files['file'].save(target_image_name)
    im0 = cv2.imread(target_image_name)  # 打开图片
    resize_scale = output_size / im0.shape[0]
    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
    cv2.imwrite("images/show.jpg", im0)
    return json.dumps({"data": None})


@app.route('/show/<path:filename>', methods=['GET', 'POST'])
def get_show_picture(filename):
    return send_from_directory('images', filename)


@app.route('/show/pre_show.png', methods=['GET', 'POST'])
def get_pre_show_picture():
    return send_from_directory('images', 'pre_show.png')


@app.route('/front/index.css', methods=['GET', 'POST'])
def get_front_index_css():
    return send_from_directory('templates', 'index.css')


@app.route('/front/axios.min.js', methods=['GET', 'POST'])
def get_front_axios_min():
    return send_from_directory('templates', 'axios.min.js')


@app.route('/front/index.js', methods=['GET', 'POST'])
def get_front_index_js():
    return send_from_directory('templates', 'index.js')


@app.route('/front/vue.js', methods=['GET', 'POST'])
def get_front_vue():
    return send_from_directory('templates', 'vue.js')


@app.route('/predict', methods=['post'])
def predict_img():
    """检测图片"""
    source = 'images/show.jpg'  # file/dir/URL/glob, 0 for webcam
    imgsz = [480, 480]  # inference size (pixels)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    view_img = False  # show results
    save_txt = False  # save results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # ugmented inference
    visualize = False  # visualize features
    line_thickness = 2  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    print(source)
    source = str(source)
    webcam = False
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                        #                  BGR=True)
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            # Stream results
            im0 = annotator.result()
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond
            # Save results (image with detections)
            resize_scale = output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/result.jpg", im0)
            return json.dumps({"data": None})


@app.route('/show/result.jpg', methods=['GET', 'POST'])
def get_result_picture():
    return send_from_directory('images', 'result.jpg')


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


if __name__ == '__main__':
    # app.debug = False
    app.run(host='0.0.0.0', port=5003)
