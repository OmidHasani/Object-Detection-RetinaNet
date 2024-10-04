### Overall Process Summary:

1. **Data Acquisition**: Download and extract the dataset from a specified URL, preparing it for use in training and evaluation.

2. **Bounding Box Manipulation**: Implement functions to swap coordinates, convert bounding box formats, and compute Intersection over Union (IoU) for matching ground truth boxes with anchor boxes.

3. **Image Preprocessing**: Apply various preprocessing techniques to images, including resizing, padding, and random horizontal flipping to enhance data variability.

4. **Anchor Box Generation**: Create anchor boxes of different scales and aspect ratios that serve as reference points for detecting objects in the images.

5. **Model Architecture**: Build the RetinaNet model using a backbone network (ResNet50) and a Feature Pyramid Network (FPN) to handle multi-scale object detection.

6. **Loss Function Definition**: Implement custom loss functions for classification and bounding box regression to effectively train the model.

7. **Model Training**: Compile the model and train it using the prepared dataset, employing techniques such as batch processing and validation.

8. **Inference and Visualization**: After training, load the model weights and perform inference on test images, decoding the predictions and visualizing the detected objects with bounding boxes and class labels.

### In Summary:
The project successfully demonstrates the implementation of a state-of-the-art object detection system, capable of identifying and localizing various objects within images through a structured and systematic approach.
---
### Process of this project : 

### 1. **Downloading and Extracting Data**
At the start, data is downloaded and extracted from a zip file:

```python
url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)

with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")
```
The dataset is downloaded from a specified URL and saved as `data.zip`. This file is then extracted into the current working directory.

### 2. **Bounding Box Operations**
Several utility functions are defined to manipulate bounding boxes:
- **`swap_xy(boxes)`**: Swaps the x and y coordinates of the bounding boxes.
- **`convert_to_xywh(boxes)`**: Converts the bounding box format to (center, width, height).
- **`compute_iou(boxes1, boxes2)`**: Computes the **Intersection over Union (IoU)** between two sets of bounding boxes.

### 3. **Visualizing Detections**
The `visualize_detections` function visualizes the predicted bounding boxes, class labels, and confidence scores on an image:

```python
def visualize_detections(image, boxes, classes, scores):
    # Visualizes detected objects on an image
```
This function draws the bounding boxes on the image and displays the class name with its confidence score.

### 4. **Anchor Box Generation**
The **AnchorBox** class generates anchor boxes at multiple scales and aspect ratios, which are used to predict potential object locations:

```python
class AnchorBox:
    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self._areas = [32, 64, 128, 256, 512]  # Sizes of anchor boxes
```
These anchor boxes are used by the model to make predictions across different spatial locations of an image.

### 5. **Preprocessing and Data Augmentation**
The function `preprocess_data` handles image preprocessing. It:
- Swaps the x and y coordinates of the bounding boxes.
- Flips the image horizontally with a 50% probability.
- Resizes and pads the image while preserving its aspect ratio.
- Converts bounding boxes to the (center, width, height) format.

```python
def preprocess_data(sample):
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)
    return image, bbox, class_id
```

### 6. **Label Encoding**
The **LabelEncoder** class is responsible for encoding ground truth data (bounding boxes and class labels) into a format suitable for training:

```python
class LabelEncoder:
    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        # Encodes batch of images and corresponding labels
```
It matches anchor boxes to ground truth boxes and computes the target for training.

### 7. **Backbone Network (ResNet50)**
The **ResNet50** backbone is used to extract features from the images:

```python
def get_backbone():
    backbone = keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])
    return keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])
```
The `c3_output`, `c4_output`, and `c5_output` feature maps from different layers of ResNet50 are used to create a **Feature Pyramid**.

### 8. **Feature Pyramid Network (FPN)**
The **FeaturePyramid** class combines features from different layers of the backbone to handle multi-scale object detection:

```python
class FeaturePyramid(keras.layers.Layer):
    def call(self, images, training=False):
        # Builds the feature pyramid with feature maps from the backbone
```
It upsamples the deeper layers and adds them to the corresponding layers in the pyramid to provide higher resolution features.

### 9. **RetinaNet Model**
The **RetinaNet** class implements the model architecture that combines the feature pyramid network and the classification and regression heads:

```python
class RetinaNet(keras.Model):
    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        # Reshaping outputs from the heads for predictions
```
- **Classification Head**: This predicts the class probabilities for each anchor box.
- **Box Regression Head**: This predicts the adjustments needed to refine the anchor boxes to better fit the objects.

### 10. **Decoding Predictions**
The **DecodePredictions** class processes the outputs from the RetinaNet model to convert them back into a usable format. This involves decoding bounding box predictions and applying non-maximum suppression (NMS) to filter out duplicate detections:

```python
class DecodePredictions(tf.keras.layers.Layer):
    def call(self, images, predictions):
        # Decode box predictions and apply NMS
```

### 11. **Loss Functions**
- **`RetinaNetBoxLoss`**: Implements Smooth L1 loss for bounding box regression.
- **`RetinaNetClassificationLoss`**: Implements Focal Loss for addressing class imbalance in the dataset.
- **`RetinaNetLoss`**: A wrapper that combines both losses to calculate the overall loss during training:

```python
class RetinaNetLoss(tf.losses.Loss):
    def call(self, y_true, y_pred):
        # Combine both classification and box losses
```

### 12. **Model Compilation and Training**
The model is compiled with the defined loss function and the SGD optimizer, and then trained on the COCO dataset:

```python
model.compile(loss=loss_fn, optimizer=optimizer)

# Loading the COCO dataset
(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

# Training the model with defined callbacks
model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
```
### 13. **Model Inference**
After training, the model is prepared for inference by loading the latest weights and making predictions on input images:

```python
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

# Read and preprocess a test image
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)
```

### 14. **Image Preparation and Visualization**
Finally, a function is defined to prepare the input image, and the predicted bounding boxes along with class names and scores are visualized using the previously defined `visualize_detections` function:

```python
image = tf.cast(image12, dtype=tf.float32)
input_image, ratio = prepare_image(image)
detections = inference_model.predict(input_image)
# Visualizing the detections
visualize_detections(
    image,
    detections.nmsed_boxes[0][:num_detections] / ratio,
    class_names,
    detections.nmsed_scores[0][:num_detections],
)
```

---

### Output of this project : 

![objectgit](https://github.com/user-attachments/assets/1ac3ab86-321a-4e11-bce6-aa61f3d78973)
