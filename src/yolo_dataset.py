from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Callable
import json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms



class ClassMap:
    """
    Mapeia IDs de classe para nomes e vice-versa.
    Espera um arquivo de texto com um nome de classe por linha.
    """
    def __init__(self, classes_path: str | Path) -> None:
        self.classes_path = Path(classes_path)
        names = []
        for line in self.classes_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                names.append(name)
        self.id2label = {i: n for i, n in enumerate(names)}
        self.label2id = {n: i for i, n in enumerate(names)}

    def name(self, class_id: int) -> str:
        return self.id2label.get(class_id, str(class_id))



class YoloTxtParser:
    """
    Espera linhas no formato YOLO:
      class_id x_center y_center w h
    Retorna bbox em xywh com (x,y) no topo-esquerdo, normalizado.
    """
    def parse(self, label_path: str | Path) -> tuple[list[int], list[list[float]]]:
        label_path = Path(label_path)
        ids: list[int] = []
        bboxes: list[list[float]] = []

        if not label_path.exists():
            return ids, bboxes

        for raw in label_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1]); yc = float(parts[2])
                w = float(parts[3]);  h = float(parts[4])
            except ValueError:
                continue

            x = xc - w / 2.0
            y = yc - h / 2.0

            ids.append(cls)
            bboxes.append([x, y, w, h])

        return ids, bboxes



@dataclass(frozen=True)
class ManifestItem:
    image_path: str
    label_path: Optional[str]
    rel_path: str
    split: Optional[str] = None


class ManifestDataset(Dataset):
    """
    Lê artifacts/dataset_index.jsonl e fornece amostras no formato:
      {
        "image": PIL.Image,
        "width": int,
        "height": int,
        "objects": {"id": [...], "category": [...], "bbox": [[x,y,w,h], ...]}
      }
    """
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        classes_path: str | Path,
        yolo_parser: Optional[YoloTxtParser] = None,
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.class_map = ClassMap(classes_path)
        self.parser = yolo_parser or YoloTxtParser()
        self.transform = transform
        self.to_tensor = tv_transforms.ToTensor()
        self.items = self._load_items()

    def _load_items(self) -> list[ManifestItem]:
        items: list[ManifestItem] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("split") != self.split:
                    continue
                items.append(
                    ManifestItem(
                        image_path=obj["image_path"],
                        label_path=obj.get("label_path"),
                        rel_path=obj.get("rel_path", ""),
                        split=obj.get("split"),
                    )
                )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        img = Image.open(it.image_path).convert("RGB")
        width, height = img.size

        ids: list[int] = []
        bboxes: list[list[float]] = []

        if it.label_path:
            ids, bboxes = self.parser.parse(it.label_path)

        categories = [self.class_map.name(i) for i in ids]
        
        image_data: Any = img
        if self.transform:
            try:
                transformed = self.transform(image=np.array(img), bboxes=bboxes, labels=ids)
            except TypeError:
                transformed = self.transform(img)

            if isinstance(transformed, dict):
                image_data = transformed.get("image", img)
                bboxes = transformed.get("bboxes", bboxes)
                ids = transformed.get("labels", ids)
                categories = [self.class_map.name(i) for i in ids]
            else:
                image_data = transformed

        if isinstance(image_data, Image.Image):
            tensor_image = self.to_tensor(image_data)
        elif isinstance(image_data, torch.Tensor):
            tensor_image = image_data
        else:
            arr = np.asarray(image_data)
            tensor_image = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        return {
            "image": tensor_image,
            "width": width,
            "height": height,
            "rel_path": it.rel_path,
            "objects": {
                "id": ids,
                "category": categories,
                "bbox": bboxes,  # xywh (normalizado)
            },
        }



class DatasetVisualizer:
    def draw(self, dataset: ManifestDataset, idx: int) -> Image.Image:
        sample = dataset[idx]
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = tv_transforms.ToPILImage()(image)
        image = image.copy()
        ann = sample["objects"]
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for i in range(len(ann["id"])):
            x, y, w, h = ann["bbox"][i]
            # Se for normalizado, converte para pixels
            if max(x, y, w, h) <= 1.0:
                x1 = int(x * width)
                y1 = int(y * height)
                x2 = int((x + w) * width)
                y2 = int((y + h) * height)
            else:
                x1 = int(x); y1 = int(y)
                x2 = int(x + w); y2 = int(y + h)

            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            draw.text((x1, y1), ann["category"][i], fill="white")

        return image

    def plot_grid(self, dataset: ManifestDataset, indices, cols: int = 3, figsize=(15, 10)) -> None:
        indices = list(indices)
        n = len(indices)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i in range(rows * cols):
            r = i // cols
            c = i % cols
            ax = axes[r][c]

            if i < n:
                img = self.draw(dataset, indices[i])
                ax.imshow(img)
                ax.set_title(f"idx={indices[i]}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
