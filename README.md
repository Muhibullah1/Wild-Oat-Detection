# Wild-Oat-Detection

This repository implements a geometry-aware wild oat detection framework for field imagery. The pipeline includes three steps: 
1. Semantic segmentation of the full image
2. Crop row detection to identify inter-row regions
3. Post-processing to classify oats within inter-row areas as wild oat

# Segmentation
```
seg_model = resnet50_segnet(n_classes=2, input_height=1088, input_width=1440)
seg_model.load_weights('2023-09-29-Narrow_Leaf/2023-09-29-Narrow_Leaf.1')
path = '/home'
images = glob.glob( path + '/images/*.jpg')
pred_path = base_path+'/preds'
Pred  = seg_model.predict_multiple(inp_dir = path, out_dir = pred_path)
```
## Load weights
```
RowModel = CropRowDetectionModel(3,1) # 3 input channels (RGB) and 1 output channel
RowModel.load_state_dict(torch.load('./best_row.pt'))
```
## Row Detection
```
path_rows =   path + "/rows/" # Saves the row coordinates
os.makedirs(path_rows, exist_ok=True)
for input_img in tqdm(images):
    op_image_name = input_img.split('/')[-1]
    rows, average_row_width = predict(RowModel, images, path_rows, device)
```
## Post Processing
```
from WildOats import process_segmentation_predictions, create_output_directories, overlay_images, process_images
image_paths, mask_paths, row_paths, width_paths, image_formats = [], [], [], [], ['*.jpg', '*.jpeg', '*.png', '*.JPG']
for fmt in image_formats:
    image_paths.extend(sorted(glob.glob(os.path.join(base_path, 'images', fmt))))
for img_path in tqdm(image_paths):
    base_name = os.path.splitext(os.path.basename(images))[0]
    mask_path = os.path.join(base_path, masks_path, f'{base_name}.png')
    text_path = os.path.join(base_path, row_path, f'{base_name}.txt')
    width_path = os.path.join(base_path, width, f'{base_name}.txt')
    mask_paths.append(mask_path)
    text_paths.append(text_path)
    width_paths.append(width_path)
main.process_images(images, mask_paths, text_paths, width_paths, base_path)
```
