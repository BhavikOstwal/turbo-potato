import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import argparse
import time
import torch
from pathlib import Path

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages


seg_model = tf.keras.models.load_model('seg.h5')
yolo_model = YOLO("best.pt")


def run_inference(frame, model_name):
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if model_name == "yolo":
        result = yolo_model(frame)
        rendered_frame = result[0].plot()

    elif model_name == "seg":
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((seg_model.input_shape[2], seg_model.input_shape[1]))

        # Convert to NumPy array and normalize
        img_array = np.array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # Run the model to get predictions
        predictions = seg_model.predict(img_input)

        # Get the segmentation mask
        mask = np.argmax(predictions[0], axis=-1).astype(np.uint8)

        # Resize the mask back to the original frame size
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply color map for visualization
        rendered_frame = cv2.applyColorMap(mask_resized * 25, cv2.COLORMAP_JET)

    return rendered_frame


def process_video(source, model):
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = run_inference(frame, model)
        cv2.imshow(f"{model} Output", processed_frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(source, model):
    # Read the image
    image = cv2.imread(source)
    if image is None:
        print(f"Failed to load image: {source}")
        return

    # Process and display the image
    processed_image = run_inference(image, model)
        
    # Show the processed image
    cv2.imshow(f"{model} Output", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Processing complete!")
    


def detect2(sor):
    # setting and directories
    source, weights, imgsz = sor, "drivable.pt", 640

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        [pred, anchor_grid], seg, ll = model(img)

        # Apply NMS
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, 0.3, 0.45, classes=args.classes, agnostic=args.agnostic_nms)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for det in pred:  # detections per image
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=3)

            # Show segmentation results
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # *Display the video frame*
            cv2.imshow('Drivable Inference', im0)

            # *Press 'q' to exit*
            if args.mode == 'img':
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    print(f'Done. ({time.time() - t0:.3f}s)')

    # *Release resources*
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image or video with a selected model.")
    parser.add_argument("--mode", type=str, choices=["img", "video"], required=True, help="Choose 'img' for image or 'video' for video.")
    parser.add_argument("--source", type=str, required=True, help="Path to the image or video source (or camera index for video).")
    parser.add_argument("--model", type=str, choices=["yolo", "seg", "drivable"], required=True, help="Specify the model to use (e.g., 'YOLO', 'seg').")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    
    args = parser.parse_args()

    if args.model == "drivable":
        detect2(args.source)
    else:
        if args.mode == "img":
            process_image(args.source, args.model)
        elif args.mode == "video":
            process_video(int(args.source) if args.source.isdigit() else args.source, args.model)

    print("Finished processing.")

