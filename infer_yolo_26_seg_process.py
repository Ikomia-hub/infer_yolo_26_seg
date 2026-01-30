import copy
import os

import torch

from ikomia import core, dataprocess, utils
import cv2

from ultralytics import YOLO
from ultralytics import download


class InferYolo26SegParam(core.CWorkflowTaskParam):
    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolo26m-seg"
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.7
        self.update = False
        self.model_weight_file = ""

    def set_values(self, param_map):
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.update = True

    def get_values(self):
        param_map = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "input_size": str(self.input_size),
            "conf_thres": str(self.conf_thres),
            "iou_thres": str(self.iou_thres),
            "model_weight_file": str(self.model_weight_file)
        }
        return param_map


class InferYolo26Seg(dataprocess.CInstanceSegmentationTask):
    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)

        if param is None:
            self.set_param_object(InferYolo26SegParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.repo = "ultralytics/assets"
        self.version = "v8.4.0"
        self.device = torch.device("cpu")
        self.classes = None
        self.model = None
        self.half = False

    def get_progress_steps(self):
        return 1

    def resize_to_stride(self, image, imgsz, stride=32):
        scale_factor = imgsz / max(image.shape[:2])
        target_width = int(scale_factor * image.shape[1])
        target_height = int(scale_factor * image.shape[0])

        new_width = ((target_width + stride - 1) // stride) * stride
        new_height = ((target_height + stride - 1) // stride) * stride

        dw = image.shape[1] / new_width
        dh = image.shape[0] / new_height

        resized_image = cv2.resize(image, (new_width, new_height))

        return resized_image, dw, dh

    def _load_model(self):
        param = self.get_param_object()
        self.device = torch.device(
            "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        self.half = True if param.cuda and torch.cuda.is_available() else False

        if param.model_weight_file:
            self.model = YOLO(param.model_weight_file)
        else:
            model_folder = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "weights")
            os.makedirs(model_folder, exist_ok=True)
            model_weights = os.path.join(
                str(model_folder), f"{param.model_name}.pt")

            if not os.path.isfile(model_weights):
                url = f"https://github.com/{self.repo}/releases/download/{self.version}/{param.model_name}.pt"
                download(url=url, dir=model_folder, unzip=True)

            self.model = YOLO(model_weights)

        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        self.begin_task_run()

        self.get_output(1).clear_data()

        param = self.get_param_object()

        img_input = self.get_input(0)
        ini_src_image = img_input.get_image()

        src_image, dw, dh = self.resize_to_stride(
            image=ini_src_image,
            imgsz=param.input_size
        )

        results = self.model.predict(
            src_image,
            save=False,
            imgsz=param.input_size,
            conf=param.conf_thres,
            iou=param.iou_thres,
            half=self.half,
            device=self.device
        )

        self.classes = list(results[0].names.values())
        self.set_names(self.classes)

        if results[0].masks is not None:
            boxes = results[0].boxes.xyxy
            confidences = results[0].boxes.conf
            class_idx = results[0].boxes.cls
            masks = results[0].masks.data
            masks = masks.detach().cpu().numpy()

            for i, (box, conf, cls, mask) in enumerate(zip(boxes, confidences, class_idx, masks)):
                box = box.detach().cpu().numpy()
                mask = cv2.resize(mask, ini_src_image.shape[:2][::-1])
                x1 = box[0] * dw
                x2 = box[2] * dw
                y1 = box[1] * dh
                y2 = box[3] * dh
                width = x2 - x1
                height = y2 - y1
                self.add_object(
                    i,
                    0,
                    int(cls),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(width),
                    float(height),
                    mask
                )

        self.emit_step_progress()
        self.end_task_run()


class InferYolo26SegFactory(dataprocess.CTaskFactory):
    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        self.info.name = "infer_yolo_26_seg"
        self.info.short_description = "Inference with YOLO26 segmentation models"
        self.info.path = "Plugins/Python/Instance Segmentation"
        self.info.version = "1.0.0"
        self.min_ikomia_version = "0.15.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2026
        self.info.license = "AGPL-3.0"
        self.info.documentation_link = "https://docs.ultralytics.com/"
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_26_seg"
        self.info.original_repository = "https://github.com/ultralytics/ultralytics"
        self.info.keywords = "YOLO, YOLO26, instance, segmentation, ultralytics"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INSTANCE_SEGMENTATION"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        return InferYolo26Seg(self.info.name, param)
