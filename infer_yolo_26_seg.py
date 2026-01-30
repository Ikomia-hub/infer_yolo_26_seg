"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from infer_yolo_26_seg.infer_yolo_26_seg_process import InferYolo26SegFactory
from infer_yolo_26_seg.infer_yolo_26_seg_process import InferYolo26SegParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return InferYolo26SegFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_yolo_26_seg.infer_yolo_26_seg_widget import InferYolo26SegWidgetFactory
        return InferYolo26SegWidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return InferYolo26SegParamFactory()
