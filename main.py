import cv2
import numpy as np
import pyzed.sl as sl
from PIL import Image
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
import torch
import threading
import time

class RealTimePeopleSegmenter:
    def __init__(self, model_path, inpainting_device):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.inpainting_device = inpainting_device
        self.simple_lama = SimpleLama(device=self.inpainting_device)

    def _segment_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
        results = self.model(img)
        H, W, _ = img.shape
        final_mask = np.zeros((H, W), dtype=np.uint8)

        for result in results:
            for j, mask in enumerate(result.masks.data):
                cls_id = result.boxes.cls[j]
                cls_name = self.names[int(cls_id)]
                if cls_name == 'person':
                    mask = mask.cpu().numpy() * 255
                    mask = cv2.resize(mask, (W, H))
                    mask = mask.astype(np.uint8)
                    final_mask[mask > 0] = 255

        return final_mask

    def _inpaint_image(self, image, mask):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
        mask_pil = Image.fromarray(mask).convert('L')
        inpainted_image_pil = self.simple_lama(image_pil, mask_pil)
        return cv2.cvtColor(np.array(inpainted_image_pil), cv2.COLOR_RGB2BGR)

    def resize_to_match(self, src, target):
        return cv2.resize(src, (target.shape[1], target.shape[0]))

    def process_frames(self, left_frame, right_frame):
        left_mask = self._segment_image(left_frame)
        right_mask = self._segment_image(right_frame)

        left_mask = left_mask.astype(np.uint8)
        right_mask = right_mask.astype(np.uint8)

        left_mask[left_mask > 0] = 255
        right_mask[right_mask > 0] = 255

        left_mask_dilated = cv2.dilate(left_mask, np.ones((17, 17), np.uint8), iterations=16)
        right_mask_dilated = cv2.dilate(right_mask, np.ones((17, 17), np.uint8), iterations=16)

        inpainted_left_background = self._inpaint_image(right_frame, left_mask_dilated)
        inpainted_right_background = self._inpaint_image(left_frame, right_mask_dilated)

        inpainted_left_background = self.resize_to_match(inpainted_left_background, left_frame)
        inpainted_right_background = self.resize_to_match(inpainted_right_background, right_frame)
        left_mask_dilated = self.resize_to_match(left_mask_dilated, left_frame)
        right_mask_dilated = self.resize_to_match(right_mask_dilated, right_frame)

        left_frame_person = cv2.bitwise_and(left_frame, left_frame, mask=left_mask)
        left_mask_not = cv2.bitwise_not(left_mask)
        right_background_inpainted = self.resize_to_match(inpainted_right_background, left_frame)
        right_background_inpainted = cv2.bitwise_and(right_background_inpainted, right_background_inpainted, mask=left_mask_not)

        left_frame_person = cv2.cvtColor(left_frame_person, cv2.COLOR_BGRA2BGR)
        left_result = cv2.bitwise_or(left_frame_person, right_background_inpainted)

        right_frame_person = cv2.bitwise_and(right_frame, right_frame, mask=right_mask)
        right_mask_not = cv2.bitwise_not(right_mask)
        left_background_inpainted = self.resize_to_match(inpainted_left_background, right_frame)
        left_background_inpainted = cv2.bitwise_and(left_background_inpainted, left_background_inpainted, mask=right_mask_not)

        right_frame_person = cv2.cvtColor(right_frame_person, cv2.COLOR_BGRA2BGR)
        right_result = cv2.bitwise_or(right_frame_person, left_background_inpainted)

        return left_result, right_result

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            config[key] = value
    return config

def main():
    config = read_config('config.txt')
    svo_file_path = config['SVO_VIDEOS_FILE_PATH']

    model_path = 'yolov8l-seg.pt'
    inpainting_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmenter = RealTimePeopleSegmenter(model_path, inpainting_device)

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.svo_real_time_mode = True

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    left_mat = sl.Mat()
    right_mat = sl.Mat()

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

            left_frame = left_mat.get_data()
            right_frame = right_mat.get_data()

            left_result, right_result = segmenter.process_frames(left_frame, right_frame)

            combined_frame = cv2.hconcat([left_result, right_result])
            cv2.imshow("ZED | Segmented and Inpainted", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
