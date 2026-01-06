from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import json
import random

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetItem:
    image_path: str
    label_path: str
    rel_path: str
    split: Optional[str] = None


class DatasetIndexer:
    """
    Faz match por nome:
      frames_raw/jogo_1_frame_000020.jpg  <->  labels/jogo_1_frame_000020.txt
    Considera APENAS imagens que possuem label.
    """
    def __init__(
        self,
        images_root: str | Path,
        labels_root: str | Path,
        label_ext: str = ".txt",
    ) -> None:
        self.images_root = Path(images_root).resolve()
        self.labels_root = Path(labels_root).resolve()
        self.label_ext = label_ext

    def _iter_images(self) -> Iterable[Path]:
        for p in self.images_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p.resolve()

    def _label_stems(self) -> set[str]:
        stems: set[str] = set()
        for p in self.labels_root.rglob(f"*{self.label_ext}"):
            if p.is_file():
                stems.add(p.stem)
        return stems

    def build_index(self) -> list[DatasetItem]:
        if not self.images_root.exists():
            raise FileNotFoundError(f"images_root não existe: {self.images_root}")
        if not self.labels_root.exists():
            raise FileNotFoundError(f"labels_root não existe: {self.labels_root}")

        labeled = self._label_stems()
        if not labeled:
            raise RuntimeError(f"Nenhum label {self.label_ext} encontrado em: {self.labels_root}")

        items: list[DatasetItem] = []
        for img in self._iter_images():
            if img.stem not in labeled:
                continue


            lab = (self.labels_root / f"{img.stem}{self.label_ext}").resolve()

            if not lab.exists():
                continue

            rel = img.relative_to(self.images_root)
            items.append(
                DatasetItem(
                    image_path=str(img),
                    label_path=str(lab),
                    rel_path=str(rel).replace("\\", "/"),
                )
            )

        return items


class RandomSplitter:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def split_train_test(
        self,
        items: list[DatasetItem],
        train_ratio: float = 0.8,
    ) -> tuple[list[DatasetItem], list[DatasetItem]]:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio precisa estar entre 0 e 1.")

        rng = random.Random(self.seed)
        shuffled = items[:]
        rng.shuffle(shuffled)

        cut = int(round(len(shuffled) * train_ratio))
        train = [DatasetItem(**{**asdict(it), "split": "train"}) for it in shuffled[:cut]]
        test = [DatasetItem(**{**asdict(it), "split": "test"}) for it in shuffled[cut:]]
        return train, test


def save_jsonl(items: list[DatasetItem], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def save_split_list(items: list[DatasetItem], out_path: str | Path, use_rel_paths: bool = True) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write((it.rel_path if use_rel_paths else it.image_path) + "\n")
