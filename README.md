# TF_to_TRT
Scripts to convert TensorFlow 1, 2 and Keras trained and exported CNN models to TensorRT inference optimized engines.

**Note:** It's important to check if the trained and exported CNN doesn't have incompatible TensorFlow operations on TensorRT before trying to use these scripts. Solve first those posible incompatibilities with a configuration script using GraphSurgeon.

**More info:** https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#supported-ops

## 1. Requisites for Jetpack 4.3 - NVIDIA Jetson AGX Xavier
* CUDA
* TensorFlow 1 and 2
* Keras
* TensorRT 6
* UFF Converter
* GraphSurgeon

## 2. Patch GraphSurgeon (TF1)
1.  Copy and replace the *node_manipulation.py* file to */usr/lib/python3.6/dist-packages/graphsurgeon/*:
    * **sudo cp -f node_manipulation.py /usr/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py**

## 3. Requisites before exporting a TF2 CNN model
1.  class DNN_model(**tf.Module**)
2.  **self.trainableVar**
3.  **@tf.function** decorator on call(self, x) function
4.  No dropout layers
5.  **tf.saved_model.save(DNN, saved_model_path, signatures=DNN.__call__.get_concrete_function(tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)))**

## 4. freeze.py: Convert TF2 models to TF1 frozen graphs
1.  Install TensorFlow 2
2.  Execute freeze.py:
    *   **python3 freeze.py path_saved_model**
3.  Get TF .pb frozen graph file

## 5. ConfigCNN.py: Convert TF1 frozen graphs to UFF files and TensorRT inference optimized engines
1.  Install TensorFlow 1
2.  Execute ConfigCNN.py:
    * **python3 ConfigCNN.py path_frozen_graph.pb**
3.  Get UFF model file (.uff) and TensorRT inference optimized engines (.bin)

# Extra: Convert SSD models
## 1. Convert original SSD+MobileNet model - ssd_inception_v2_coco(2017)
1.  Convert the SSD model on TF2 to a TF1 frozen graph with **freeze.py**
2.  Execute ConfigSSD1.py:
    * **python3 ConfigSSD1.py path_frozen_graph.pb**
3.  Get UFF model file (.uff) and TensorRT inference optimized engines (.bin)

## 2. Convert custom retrained SSD+MobileNet model - Only human faces
1.  Convert the SSD model on TF2 to a TF1 frozen graph with **freeze.py**
2.  Execute ConfigSSD2.py:
    * **python3 ConfigSSD2.py path_frozen_graph.pb**
3.  Get UFF model file (.uff) and TensorRT inference optimized engines (.bin)

# References:
* Original NVIDIA sample's GiHub repository: <a href=https://github.com/AastaNV/TRT_object_detection>AastaNV/TRT_object_detection</a>
* Original Jeroen Bédorf's tutorial: <a href=https://www.minds.ai/post/deploying-ssd-mobilenet-v2-on-the-nvidia-jetson-and-nano-platforms>Deploying SSD mobileNet V2 on the NVIDIA Jetson and Nano platforms</a>
* SSD retrained models for only faces to TensorRT: <a href=https://github.com/brokenerk/TRT-SSD-MobileNetV2>TensorRT Python Sample for a Re-Trained SSD MobileNet V2 Model</a>
