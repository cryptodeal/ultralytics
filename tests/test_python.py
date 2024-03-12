# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LINUX,
    MACOS,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    Retry,
    checks,
    is_dir_writeable,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9

MODEL = WEIGHTS_DIR / "path with spaces" / "yolov8n.pt"  # test spaces in path
CFG = "yolov8n.yaml"
SOURCE = ASSETS / "bus.jpg"
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
IS_TMP_WRITEABLE = is_dir_writeable(TMP)




def test_predict_img():
    """Test YOLO prediction on various types of image sources."""
    model = YOLO(MODEL)
    model("https://ultralytics.com/images/zidane.jpg")


   
