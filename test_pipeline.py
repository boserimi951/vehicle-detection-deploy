"""
Standalone end-to-end test — functions copied to match main.py exactly.
No import from main.py, so no risk of triggering FastAPI/session startup twice.

Run:
    conda activate Object_Detection
    python test_pipeline.py
"""

from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as ort
import cv2


# ---------------------------------------------------------------------------
# preprocess() — exact copy from main.py
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image, target_size=640):
    img = image.convert('RGB')
    orig_h, orig_w = img.size  # note: PIL .size is actually (width, height) —
                                 # names are swapped here but it still works
                                 # because resize() expects the same order back
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img = img.resize((new_h, new_w))

    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    canvas.paste(img, (0, 0))

    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr, scale


# ---------------------------------------------------------------------------
# postprocess() — exact copy from main.py
# ---------------------------------------------------------------------------
def postprocess(output, conf_threshold=0.25, iou_threshold=0.45,
                 orig_shape=None, letterbox_shape=(640, 640),
                 scale=None, pad=None):
    predictions = np.squeeze(output).T
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

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
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

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
        pad_x, pad_y = pad
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


# ---------------------------------------------------------------------------
# Run the full pipeline on a real test image
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    image_path = '/Users/mac/Documents/Deployment/Object_Detection/Inference_image/00000256.jpg'      # <-- change to your actual test image path
    model_path = '/Users/mac/Documents/Deployment/Object_Detection/best.onnx'          # <-- change if your onnx file has a different name

    image = Image.open(image_path)
    print(f'Loaded image: {image_path}, size={image.size}')

    img_tensor, scale = preprocess(image)
    print(f'Preprocessed. scale={scale}, tensor shape={img_tensor.shape}')

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_tensor})[0]
    print(f'Raw output shape: {output.shape}')

    detections = postprocess(output, conf_threshold= 0.01, scale=scale, pad=(0, 0))
    print(f'\nDetections found: {len(detections)}')
    for i, det in enumerate(detections):
        print(f"  [{i}] class_id={det['class_id']}  "
              f"confidence={det['confidence']:.3f}  "
              f"bbox={[round(v, 1) for v in det['bbox']]}")

    if detections:
        draw_img = image.convert('RGB').copy()
        draw = ImageDraw.Draw(draw_img)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, max(0, y1 - 12)),
                       f"{det['class_id']} {det['confidence']:.2f}",
                       fill='red')
        out_path = 'test_output.jpg'
        draw_img.save(out_path)
        print(f'\nSaved visualized result to {out_path} — open it to check the boxes look right.')
    else:
        print('\nNo detections above threshold — try lowering conf_threshold or check your test image.')
