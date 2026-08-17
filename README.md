# Traffic & Vehicle Object Detection — Deployment Pipeline

A YOLOv8-based object detection system for traffic/CCTV imagery, deployed via ONNX export and served through a FastAPI endpoint.

## Overview

- **Model**: YOLOv8, exported to ONNX for lightweight, framework-independent inference
- **Classes (11)**: pickup_truck, car, articulated_truck, bus, motorized_vehicle, work_van, single_unit_truck, pedestrian, bicycle, non-motorized_vehicle, motorcycle
- **Serving**: FastAPI `/predict` endpoint accepting image uploads, returning bounding boxes with class labels and confidence scores
- **Pipeline**: preprocess (letterbox resize/pad) → ONNX inference → postprocess (confidence threshold, NMS, coordinate rescaling to original image space)

## Model Limitations & Future Improvements

The current model was trained on a class-imbalanced, low-resolution CCTV traffic dataset, which produces two measurable effects:

- **Classification confidence** is strong for well-represented classes (e.g., cars at 0.94 confidence in testing) but weak for underrepresented ones
- **Bounding box localization** is imprecise due to inconsistent annotation quality in the source data

Given more time, I would prioritize:

1. **Rebalance the data** — audit per-class sample counts and rebalance or reweight underrepresented classes
2. **Validate annotation quality** — re-label a sample subset and check inter-annotator agreement
3. **Benchmark externally** — compare against a public dataset like BDD100K or MIO-TCD to isolate whether performance gaps stem from data quality or model architecture

These are data-centric improvements — the deployment pipeline (ONNX export, FastAPI serving, containerization) is independent of model quality and would carry over unchanged to an improved model.

## Deployment Pipeline Progress

- [x] Rung 1 — Export model to ONNX
- [x] Rung 2 — FastAPI serving with full pre/postprocessing
- [ ] Rung 3 — Containerization (Docker)
- [ ] Rung 4 — Cloud deployment
- [ ] Rung 5 — Monitoring & logging (request/response logging, basic error tracking, latency checks on the deployed endpoint)
- [ ] Rung 6 — CI/CD (automated build + test on push, so deployments aren't manual)
- [ ] Rung 7 — Documentation & polish (finalize README, demo instructions, clean up code/comments for portfolio presentation)

## Tech Stack

- YOLOv8 (Ultralytics)
- ONNX Runtime
- FastAPI
- OpenCV (NMS)
- NumPy, Pillow
