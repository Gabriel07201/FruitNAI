# FruitNinja AI
Versão em Portugês: [README.md](README.md).

Bot for **Fruit Ninja** trained with few labels, short training time, and high accuracy.
The goal of this project is to be **reproducible** and **trainable**: you can train your own detector, export it to ONNX, and plug it into the gameplay loop with clear configuration knobs (max detection age, minimum action interval, bomb safety radius, slice overshoot, etc.).

---

The initial model was trained on 70 images for 50 epochs, which took 01:30 minutes. It could perfectly distinguish between fruits and bombs, but it ended up producing many false positives on fruit fragments, which crowded the slice queue and reduced the bot’s efficiency. It also required using a lower confidence threshold because some fruits rotate and the model would get more “confused” (the banana is a good example).

The second version was trained on 124 images for 80 epochs, which made it possible to increase the confidence threshold and reduced false-positive cases. It also learned fruit rotations and even generalized to other game modes that contain the same fruits but with different visual characteristics (borders, effects, etc.).

The third model was trained on 190 images for 120 epochs, which brought the same improvements seen in the second model, but with higher accuracy.

---

## Key Features

- Real-time detection with low latency.
- Inference using **ONNXRuntime** (good portability and performance).
- Action quality control:
  - **Detection age filter**.
  - **Minimum interval between actions**.
- Slice heuristics:
  - **Overshoot** based on bbox diagonal.
  - “Segment” correction for bomb recheck.
- Anti-bomb heuristics:
  - “Safe zone” based on the bomb bbox diagonal.
  - Instant recheck in critical regions.
- Optional diagnostics: logs and overlay (when enabled).

---

## How it works (high-level)

The bot runs a main loop roughly like this:
1. **Capture** a cropped region of the game window.
2. **Pre-process** with letterbox/resize while preserving aspect ratio.
3. **Run inference** on the model.
4. **Post-process**:
   - convert outputs into bboxes in the original scale
   - split fruits and bombs by class
   - apply confidence thresholds
5. **Decide action**:
   - ignore stale detections
   - choose fruit targets (priority by confidence/area/position)
   - compute the slice trajectory
   - validate safety against bombs (safe zone + rechecks)
6. **Execute the slice** via the controller.

---

## Running the bot

Create a conda environment (or activate an existing one):

```bash
# create environment in cmd
conda env create -f environment.yml
# activate environment
conda activate "environment_name"
```
After activating, install dependencies:
```bash
pip install .
pip install -r requirements.txt
```

Open Fruit Ninja via the Play Store launcher (Google Play Games).
It is recommended to reduce game quality in the launcher settings and also reduce the window size to improve inference performance.

Then run with the active environment:
```bash
fruitnai-bot
```
Or run directly as a module:
Default inference (no debug/logs):
```bash
python -m bot.run
```
With diagnostics (draw/debug/logs):
```bash
python -m bot.run --enable-diagnostics
```

## Useful configuration (tuning)

Main parameters live at the top of `src/bot/run.py`, with defaults tuned for:

**Detection/time**

  - `MAX_DET_AGE_S`: maximum detection age (avoids slicing “in the past”)

  - `MIN_ACTION_INTERVAL_S`: minimum interval between slices

  - `RECENT_TTL_S` / `RECENT_RADIUS_PX`: temporary region cooldown

**Inference**

  - `PREDICT_IMGSZ`: model input image size

  - `MIN_FRUIT_CONF` / `MIN_FRUIT_AREA`: confidence/area filtering

**Slice heuristics**

  - `OVERSHOOT_BASE_PX` / `OVERSHOOT_DIAG_FACTOR`: slice overshoot

  - `SEGMENT_OFFSET_PX`: offset for instant recheck

**Anti-bomb heuristics**

  - `BOMB_SAFE_BASE_PX` / `BOMB_SAFE_DIAG_FACTOR`: safe zone

  - `INSTANT_SAFE_BASE_PX`: instant recheck radius

To switch the ONNX model, update:

  - `ONNX_PATH = "models/runs/fruitninja_yolo11n3/weights/best.onnx"`

## Theory

A common challenge in modern ML SOTA is the massive size of models, which increases training time and cost.
This project focuses on fine-tuning a pre-trained model to perform domain adaptation for the game.

**Fine-tuning** is part of **Transfer Learning**: it takes a model trained by someone else and trains (or replaces + trains) the final layers responsible for classification or other tasks.
Techniques like **data augmentation** can improve fine-tuning by increasing dataset variability through image transformations, reducing overfitting and encouraging the model to learn task-relevant features.

Other training techniques also exist (adding convolutional layers, concatenating layers, etc.).

### Choosing the base model (YOLO)

YOLO focuses on object detection, so it predicts both class and coordinates.
To understand YOLO, it helps to mention some earlier architectures:

R-CNN: uses Selective Search to propose regions, then a CNN to classify each region.
![alt text](imgs/rcnn.png)

Fast R-CNN: trains in a single stage (instead of multiple) and uses ROI Pooling.
![alt text](imgs/fasrcnn.png)

Faster R-CNN: removes Selective Search and introduces the **RPN** (Region Proposal Network), improving inference time.
![alt text](imgs/faster_rcnn.png)

FPN (Feature Pyramid Network): builds multi-scale features by downscaling and upscaling representations to improve detection.
![alt text](imgs/fpn.png)

**ALL IMAGES WERE TAKEN FROM HUGGING FACE**

Now, YOLO:
YOLO is a real-time object detector using a single network. Before YOLO, pipelines often relied on image classifiers applied over many regions, which was slower due to multiple components.

YOLO is a single-shot detector: bounding boxes and classes are predicted in one pass, making it fast.

It divides the image into an SxS grid. B is the number of boxes predicted per cell. The cell containing the object’s center is responsible for predicting it. A confidence score is computed, using Intersection over Union (IoU) between the predicted and the ground-truth box.
Boxes are represented by 4 numbers (x, y, w, h), where (x, y) is the center and (w, h) are dimensions.

Architecturally, it is a CNN inspired by GoogLeNet, using 1x1 convolutions to reduce feature map depth. It uses LeakyReLU and a Linear final layer.
![alt text](imgs/yolo_arch.png)

Finally, many boxes are produced and a score-based filtering step keeps only the most confident ones. The general process can be seen below:
![alt text](imgs/object-detection-gif.gif)

Since this project uses YOLOv8, here is a high-level summary of key changes across versions:

- v2: backbone changed to Darknet-19 with batch normalization; introduced anchor boxes (e.g., 5 anchors on a 13x13 grid → 13x13x5 = 845 boxes).

- v3: changed to Darknet-53; used 3 grid scales (13x13, 26x26, 52x52) increasing boxes (13x13x3 + 26x26x3 + 52x52x3 = 10,647).

- v4: changed to CSPDarknet53 for efficiency; introduced several training-time improvements (bag-of-freebies).

- v5: added auto-anchors (KNN-based) to adapt anchors automatically.

- v6: introduced re-parameterization concepts (RepVGG), and new loss functions (e.g., Varifocal loss, IoU-family losses, Distribution Focal).

- v7: improved bag-of-freebies and label assignment strategies; proposed scaling methods.

- v8: changed architecture and moved to an anchor-free approach.

## Training your own model

This repo includes a simple pipeline to build a dataset and train:

1. **Dataset indexing**

   - `src/indexer.py` creates `dataset_index.jsonl` from images + labels.

2. **Ultralytics dataset build**

   - `src/ultralytics_builder.py` converts the manifest into `data.yaml`.

3. **Train / export**

   - Notebooks in `notebooks/` are numbered in execution order.

   - They include **export to ONNX** steps.

4. **Extra**

   - If you want a more “hardcoded” training implementation, see the notebook `extra_train_manual`, where an additional model is trained “by hand” using Hugging Face **transformers**.

> Tip: keep YOLO label format (class_id x_center y_center w h).

### Step-by-step

1. Record a gameplay video using any recorder.

2. Extract frames from the video

    - Recommended: ffmpeg
    ```bash
    # detect where the game pixels are in the video
    ffmpeg -i "your_video.ext" -vf "cropdetect=24:16:0" -t 10 -f null -
    # extract frames using the crop
    ffmpeg -i "your_video.ext" -vf "crop=w:h:x:y,fps=5" -q:v 2 "frames/frame_%06d.jpg"
    ```

3. Label the images

    - LabelImg (or any tool you prefer)

    - At least 2 classes (Fruit, Bomb)

    - Include images with only background

    - Include images with many sliced fruit fragments (to reduce false positives)

4. Follow the `notebooks/` and adapt to your preferences

5. After exporting your **ONNX** model, update `src/bot/run.py` and edit `ONNX_PATH`