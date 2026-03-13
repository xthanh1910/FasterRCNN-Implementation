from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import os
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default="trained_models/best.pt", help="Load from this checkpoint")
    parser.add_argument("--image_path", "-i", type=str, help="Path to image", required=True)
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3, help="Confident threshold")

    args = parser.parse_args()
    return args

categories = [ 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',               
                            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                            'train', 'tvmonitor' ]

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_mobilenet_v3_large_320_fpn()

    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 21
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=num_classes)

    checkpoint = torch.load(args.saved_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.float()

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))/255.0
    image = [torch.from_numpy(image).to(device).float()]

    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        bboxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > args.conf_threshold:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(ori_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                category = categories[label]        
                cv2.putText(ori_image, category, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)    
        cv2.imwrite("prediction.png", ori_image)


if __name__ == '__main__':
    args = get_args()
    test(args)