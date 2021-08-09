
from segments import SegmentsDataset
from utils import visualize, train_model
from segments import SegmentsClient # Install this package with 'pip install segments-ai'
from utils import train_model

from segments.utils import bitmap2file


# Set up the client
client = SegmentsClient('3bcb9a3ef2490a402400aa7bcf672351b4c3a8d7')
dataset_name = 'Saulzar/vines' # Name of a dataset you've created on Segments.ai

release = client.get_release(dataset_name, 'v0.2')
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by='labeled')

model = train_model(dataset, resume=False, train_scale=2.0)


# Initialize a new dataset, this time containing only unlabeled images
testing = SegmentsDataset(release, labelset='ground-truth', filter_by='unlabeled')

for sample in testing:
    # Generate label predictions
    image = sample['image']
    segmentation_bitmap, annotations = model(image)
    
    # Visualize the predictions
    # visualize(image, segmentation_bitmap)
    # print(annotations)
    
    # Upload the predictions to Segments.ai
    file = bitmap2file(segmentation_bitmap)
    asset = client.upload_asset(file, 'label.png')    
    attributes = {
        'format_version': '0.1',
        'annotations': annotations,
        'segmentation_bitmap': { 'url': asset['url'] },
    }
    client.add_label(sample['uuid'], 'ground-truth', attributes, label_status='PRELABELED')
