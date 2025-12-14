from rfdetr import RFDETRBase
from torchinfo import summary
# Experiment 2: Density Init Only (No SOQB)
# This was our previous "Best Baseline"
model = RFDETRBase()

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/UCAS_AOD_COCO',
    coco_path='/home/fyb/datasets/UCAS_AOD_COCO',
    epochs=1,
    batch_size=6,
    grad_accum_steps=1,
    lr=1e-4,
    num_workers=2,
    output_dir='results/2basealine_UCAS_AOD_COCO',
)   