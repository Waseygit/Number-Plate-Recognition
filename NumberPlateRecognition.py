import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ANPR:
     def __init__(self, model_path: str = "yolo11n.pt"):
        """Initializes the ANPR system."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def detect_plates(self, im0: np.ndarray):
        """Detects license plates in a image."""
        results = self.model.predict(im0, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        return boxes

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        """Performs OCR on the cropped license plate region."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = im0[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = self.reader.readtext(gray, detail=0, paragraph=True)
        return " ".join(text).strip() if text else ""

    def infer_video(self, source: str = 0, output_path: str = None, display: bool = True):
        """Performs real-time ANPR on a video stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Starting ANPR video inference... Press 'q' to quit.")

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            boxes = self.detect_plates(im0)
            ann = Annotator(im0, line_width=4)
            for bbox in boxes:
                text = self.extract_text(im0, bbox)
                ann.box_label(bbox, label=text, color=colors(17, True))

            if display:
                cv2.imshow("ANPR (Press 'q' to exit)", im0)
            if writer:
                writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    anpr = ANPR(model_path="anpr_best.pt")  # Use trained YOLO license plate model
    anpr.infer_video(source="videos/acar.mp4", output_path="anpr_output.mp4", display=True)
