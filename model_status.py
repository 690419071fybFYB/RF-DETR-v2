import sys
import os

# Insert the directory containing 'rfdetr' package to the beginning of sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from torchinfo import summary
from rfdetr import RFDETRBase
model=RFDETRBase()
summary(model.model.model, input_size=(1, 3, 560, 560),depth=7)