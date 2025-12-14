from rfdetr import RFDETRBase
from torchinfo import summary
# Experiment 2: Density Init Only (No SOQB)
# This was our previous "Best Baseline"
model = RFDETRBase()

dataset="/root/UCAS_AOD_COCO"
output_dir='results/2baseline_UCAS_AOD'

model.train(
    dataset_file='coco',
    dataset_dir=dataset,
    coco_path=dataset,
    epochs=50,
    batch_size=16,
    grad_accum_steps=2,
    lr=1e-4,
    num_workers=2,
    output_dir=output_dir,
)
