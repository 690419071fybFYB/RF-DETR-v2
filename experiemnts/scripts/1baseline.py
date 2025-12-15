from rfdetr import RFDETRBase
from torchinfo import summary
# Experiment 2: Density Init Only (No SOQB)
# This was our previous "Best Baseline"
model = RFDETRBase()

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=50,
    batch_size=12,
    grad_accum_steps=2,
    lr=1e-4,
    num_workers=2,
    output_dir='results/1baseline',
    
)
