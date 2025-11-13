import os
import sys
from pathlib import Path
import cv2
import numpy as np
import rclpy
import torch
# replace with your own paths 
train_image_folder = Path("/Users/peter/Library/CloudStorage/OneDrive-Personal/Documents/MDS_UBC/OKMarine/DepthT-main/datasets/gateseg/images/train")
train_label_folder = Path("/Users/peter/Library/CloudStorage/OneDrive-Personal/Documents/MDS_UBC/OKMarine/DepthT-main/datasets/gateseg/labels/train")

test_image_folder = Path("/Users/peter/Library/CloudStorage/OneDrive-Personal/Documents/MDS_UBC/OKMarine/DepthT-main/datasets/gateseg/images/val")
test_label_folder = Path("/Users/peter/Library/CloudStorage/OneDrive-Personal/Documents/MDS_UBC/OKMarine/DepthT-main/datasets/gateseg/labels/val/val2")

# Ensure the parent directory is in sys.path to import okmr_object_detection
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from okmr_object_detection import onnx_segmentation_detector as detector


def get_precision(pred, gt):
    try:
        tp = (pred * gt).sum().item()
        fp = (pred * (1 - gt)).sum().item()
       

        precision = tp / (tp + fp + 1e-6)
       

        return precision
    except RuntimeError:
        print(f"[WARN] Skipping sample: shape mismatch pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
        return None

def get_recall(pred, gt):
    try:
        tp = (pred * gt).sum().item()
        fn = ((1 - pred) * gt).sum().item()
       

        recall = tp / (tp + fn + 1e-6)
       

        return recall
    except RuntimeError:
        print(f"[WARN] Skipping sample: shape mismatch pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
        return None




def iou_loss(pred, gt, eps=1e-6):
    try:
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        return 1 - (intersection + eps) / (union + eps)
    except RuntimeError:
        print(f"[WARN] Skipping sample: shape mismatch pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
        return None


def dice_loss(pred, gt, eps=1e-6):
    try:
        intersection = (pred * gt).sum()
        return 1 - (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)
    except RuntimeError:
        print(f"[WARN] Skipping sample: shape mismatch pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
        return None


def get_accuracy(pred, gt):
    try:
        correct = (pred == gt).sum().item()
        total = gt.numel()
        return correct / total if total > 0 else None
    except RuntimeError:
        print(f"[WARN] Skipping sample: shape mismatch pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
        return None
    

def get_model_performance(pred_mask, gt_mask):
    """Compute IoU, Dice, and Accuracy between predicted and ground truth masks."""
    pred = torch.tensor(pred_mask, dtype=torch.float32)
    gt = torch.tensor(gt_mask, dtype=torch.float32)
    iou = 1 - iou_loss(pred, gt).item()
    dice = 1 - dice_loss(pred, gt).item()
    accuracy = get_accuracy(pred, gt)
    precision = get_precision(pred, gt)
    recall = get_recall(pred, gt)
    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

def test_model_performance(image_folder,label_folder, node):
    """
    Get average model performance on all images in a folder.
    Args: Provide folder paths containing images and corresponding labels.
    Returns: Average performance metrics across all images in the folder.
    """
    performance_metrics = []
    
    # Loop through all image files
    for image_path,test_label_file in zip(sorted(image_folder.glob("*")), sorted(label_folder.glob("*"))):
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert rgb is not None, f"Failed to read image {image_path}"
        
        img_h, img_w = rgb.shape[:2]

        pred_mask = test_inference(rgb, node) #run inference to get mask
        gt_mask, _ = seg_to_mask(test_label_file, img_h, img_w) #get ground truth mask

        metrics = get_model_performance(pred_mask, gt_mask)
        performance_metrics.append(metrics)

    avg_metrics = {}

    for key in performance_metrics[0]:
        total = 0
        count = 0
        for m in performance_metrics:
            if m[key] is not None:
                total += m[key]
                count += 1
        
        avg_metrics[key] = total / count if count > 0 else None

    return avg_metrics

def seg_to_mask(label_file, img_h=480, img_w=640, normalize=True):
    """
    Convert YOLO-seg .txt label file to a binary 0/1 mask (H x W).

    label_file: path to YOLO segmentation label .txt file
    img_h, img_w: output mask size
    normalize: True if coordinates are normalized [0,1]

    Returns:
        mask: np.ndarray (H, W), dtype=np.uint8
        labels: list of (class_id, polygon)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    labels = []

    with open(label_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue  # skip invalid line

        cls = int(parts[0])
        coords = np.array(list(map(float, parts[5:])), dtype=np.float32)
        pts = coords.reshape(-1, 2)

        if normalize:
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h

        pts = np.round(pts).astype(np.int32)

        # Fill polygon into mask
        cv2.fillPoly(mask, [pts], cls+1)  # classes start from 1 in mask, 0 is background

        labels.append((cls, pts))

    return mask, labels


def visualize_mask_on_image(image, mask, window_name="Segmentation Visualization"):
    """
    Overlay the mask on the image for visualization.
    Each class ID in `mask` gets a unique color.
    """
    # Define colors for each class
    color_map = {
        0: (0, 0, 0),   # Background – Black
        1: (0, 255, 0), # Class 1 – Green
        2: (0, 0, 255), # Class 2 – Red
        3: (255, 0, 0), # Class 3 – Blue , more can be added if needed
    }
    overlay = image.copy()

    # Apply each class color
    for class_id, color in color_map.items():
        overlay[mask == class_id] = color

    # Blend overlay with original image
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.imshow(f"{window_name}", blended)

    return blended

def visualize_prediction_and_gt(image_path, gt_label_path, node):
    """Visualize prediction and ground truth masks on the image."""
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert rgb is not None, f"Failed to read image {image_path}"
    
    img_h, img_w = rgb.shape[:2]

    pred_mask = test_inference(rgb, node) # run inference to get mask
    gt_mask, _ = seg_to_mask(gt_label_path, img_h, img_w) # get ground truth mask

    # Visualize predicted mask
    visualize_mask_on_image(rgb, pred_mask, window_name="Predicted Mask")

    # Visualize ground truth mask
    visualize_mask_on_image(rgb, gt_mask, window_name="Ground Truth Mask")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_inference(image, node):
    """Run inference """
    depth = None  # not used right now

    
    mask = node.inference(image, depth) # run inference
    return mask




def main():
    # Initialize rclpy
    rclpy.init(args=None)
    try:
   
        # Create node
        node = detector.OnnxSegmentationDetector()
        
        
        # Train model performance

        train_avg_metrics = test_model_performance(train_image_folder, train_label_folder, node)

        print(f"Train Average Metrics: {train_avg_metrics}")
        

        
        # Test model performance
        

        test_avg_metrics = test_model_performance(test_image_folder, test_label_folder, node)

        print(f"Test Average Metrics: {test_avg_metrics}")
        
        # Visualize one example
        example_image_path = test_image_folder /"22.png"
        example_gt_label_path = test_label_folder /"22.txt"
        visualize_prediction_and_gt(example_image_path, example_gt_label_path, node)

 

    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()


# rclpy.init(args=None)

# node = detector.OnnxSegmentationDetector()
# example_image_path = train_image_folder /"gate52_1.png"
# example_gt_label_path = train_label_folder /"gate52_1.txt"
# visualize_prediction_and_gt(example_image_path, example_gt_label_path, node)
