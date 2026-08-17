from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

session = ort.InferenceSession('best.onnx', providers = ['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape


def preprocess(image: Image.Image, target_size = 640):
    img = image.convert('RGB')
    orig_h, orig_w = img.size
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img = img.resize((new_h, new_w))

    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    canvas.paste(img, (0, 0))

    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis = 0)

    return arr, scale



@app.post('/predict')
async def predict (file : UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    input_tensor, scale = preprocess(image)
    output = session.run(None, {input_name: input_tensor})
    detections = postprocess(output, scale = scale, pad = (0, 0))
    return {'detections':detections}



import cv2
def postprocess(output, conf_threshold = 0.25, iou_threshold = 0.45,
                orig_shape = None, letterbox_shape = (640, 640),
                scale = None, pad = None):
    predictions = np.squeeze(output).T
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis = 1)
    confidences = np.max(class_scores, axis = 1)

    mask = confidences > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis = 1)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_threshold,
        iou_threshold
    )
    indices = np.array(indices).flatten() if len(indices) > 0 else []

    boxes_xyxy = boxes_xyxy[indices]
    class_ids = class_ids[indices]
    confidences = confidences[indices]

    if scale is not None and pad is not None:
        pad_x , pad_y = pad
        boxes_xyxy[:, [0, 2]] -= pad_x
        boxes_xyxy[:, [1, 3]] -= pad_y
        boxes_xyxy /= scale

        results = []
        for box, cls, conf in zip(boxes_xyxy, class_ids, confidences):
            results.append({
                'class_id': int(cls),
                'confidence': float(conf),
                'bbox': [float(v) for v in box]
            })
        return results


    