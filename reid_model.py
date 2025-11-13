import torch
import torchvision.transforms as T
import cv2
import numpy as np
import torchreid


class ReIDModel:
    def __init__(self, device="cuda"):
        self.device = device

        # Load OSNet_x0_25 (fast) with pretrained weights
        self.model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1000,
            pretrained=True
        ).to(self.device)

        # remove classification head - we want feature vectors only
        self.model.classifier = torch.nn.Identity()
        self.model.eval()

        # Preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def extract_batch(self, frame_bgr, boxes):
        """
        Extract features for multiple person crops in ONE forward pass.
        boxes: Nx4 array in xyxy
        """
        crops = []
        valid_ids = []

        h, w = frame_bgr.shape[:2]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # ensure valid crop
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb)
            crops.append(tensor)
            valid_ids.append(i)

        if len(crops) == 0:
            return [None] * len(boxes)

        batch = torch.stack(crops).to(self.device)

        feats = self.model(batch)
        feats = feats.cpu().numpy()

        # L2 normalize each
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10
        feats = feats / norms

        # map back to list
        output = [None] * len(boxes)
        for idx, j in enumerate(valid_ids):
            output[j] = feats[idx]

        return output
