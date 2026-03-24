from PyQt6.QtWidgets import *

from torch.cuda import is_available

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_yolo_26_seg.infer_yolo_26_seg_process import InferYolo26SegParam


class InferYolo26SegWidget(core.CWorkflowTaskWidget):
    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYolo26SegParam()
        else:
            self.parameters = param

        self.grid_layout = QGridLayout()

        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model name")
        self.combo_model.addItem("yolo26n-seg")
        self.combo_model.addItem("yolo26s-seg")
        self.combo_model.addItem("yolo26m-seg")
        self.combo_model.addItem("yolo26l-seg")
        self.combo_model.addItem("yolo26x-seg")
        self.combo_model.setCurrentText(self.parameters.model_name)

        custom_weight = bool(self.parameters.model_weight_file)
        self.check_cfg = QCheckBox("Custom model")
        self.check_cfg.setChecked(custom_weight)
        self.grid_layout.addWidget(self.check_cfg, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_cfg.stateChanged.connect(self.on_custom_weight_changed)

        self.label_hyp = QLabel("Model weight (.pt)")
        self.browse_weight_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.model_weight_file,
            tooltip="Select file",
            mode=QFileDialog.FileMode.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_weight_file, row, 1)

        self.label_hyp.setVisible(custom_weight)
        self.browse_weight_file.setVisible(custom_weight)

        self.spin_input_size = pyqtutils.append_spin(
            self.grid_layout,
            "Input size",
            self.parameters.input_size
        )

        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        self.spin_iou_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence IoU",
            self.parameters.iou_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_custom_weight_changed(self, int):
        self.label_hyp.setVisible(self.check_cfg.isChecked())
        self.browse_weight_file.setVisible(self.check_cfg.isChecked())

    def on_apply(self):
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.iou_thres = self.spin_iou_thres.value()
        if self.check_cfg.isChecked():
            self.parameters.model_weight_file = self.browse_weight_file.path
        self.parameters.update = True

        self.emit_apply(self.parameters)


class InferYolo26SegWidgetFactory(dataprocess.CWidgetFactory):
    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        self.name = "infer_yolo_26_seg"

    def create(self, param):
        return InferYolo26SegWidget(param, None)
