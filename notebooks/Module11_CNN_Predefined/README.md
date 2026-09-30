# Eight pretrained CNN notebooks

Run each independently, from top to bottom. Use Python 3.11/3.12 or Colab; internet is required for first-time downloads. GPU optional. Installation is included in every notebook.

| Notebook | Model | Use case |
|---|---|---|
| 01 | ResNet50 | Image classification |
| 02 | VGG16 | Feature map visualization |
| 03 | MobileNetV2 | Visual similarity search |
| 04 | DenseNet121 | Image clustering |
| 05 | EfficientNetB0 | Cat/dog transfer learning |
| 06 | InceptionV3 | Prediction robustness checks |
| 07 | Xception | Grad-CAM explanation |
| 08 | ConvNeXtTiny | Image novelty detection |

Data downloads automatically; no Google Drive authentication is needed. Notebooks 01/02/06/07 use a sunflower sample, and 03/04/05/08 use CIFAR-10. No external input CSV is required for these image tasks. Model weights are downloaded into the Keras cache and reused.

CIFAR-10 images are 32×32 and are enlarged for ImageNet networks. Enlargement does not recover detail. Small dataset subsets reduce runtime; they are instructional examples, not performance guarantees. These notebooks do not perform object detection or segmentation.

Validation: notebook JSON, cell structure and Python syntax were checked. TensorFlow execution, model downloads and numerical outputs were not tested in the authoring environment because TensorFlow was unavailable. Outputs are intentionally empty so the learner can run the cells. Use current compatible TensorFlow in the active kernel; if installation fails in another Python version, create a Python 3.11/3.12 environment.
