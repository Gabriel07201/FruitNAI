from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
import json
import shutil
import os


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    label_path: Optional[str]
    rel_path: str
    split: str


class UltralyticsDatasetBuilder:
    """
    Constrói um dataset no formato Ultralytics a partir do dataset_index.jsonl.

    - Mapeia split do manifest -> train/val do Ultralytics
    """
    def __init__(
        self,
        manifest_path: str | Path,
        classes_path: str | Path,
        out_dir: str | Path,
        train_split: str = "train",
        val_split: str = "test",
        mode: str = "copy",  # "copy" | "hardlink" | "symlink"
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.classes_path = Path(classes_path)
        self.out_dir = Path(out_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.mode = mode

        if self.mode not in {"copy", "hardlink", "symlink"}:
            raise ValueError("mode deve ser: copy | hardlink | symlink")

        # pastas padrão Ultralytics
        self.img_train = self.out_dir / "images" / "train"
        self.img_val = self.out_dir / "images" / "val"
        self.lab_train = self.out_dir / "labels" / "train"
        self.lab_val = self.out_dir / "labels" / "val"

    def _read_manifest(self) -> list[ManifestRow]:
        rows: list[ManifestRow] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                split = obj.get("split")
                if not split:
                    continue
                rows.append(
                    ManifestRow(
                        image_path=obj["image_path"],
                        label_path=obj.get("label_path"),
                        rel_path=obj.get("rel_path", ""),
                        split=split,
                    )
                )
        return rows

    def _read_classes(self) -> dict[int, str]:
        names = []
        for line in self.classes_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                names.append(name)
        return {i: n for i, n in enumerate(names)}

    def _ensure_dirs(self) -> None:
        for p in [self.img_train, self.img_val, self.lab_train, self.lab_val]:
            p.mkdir(parents=True, exist_ok=True)

    def _place(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            return  # idempotente

        if self.mode == "copy":
            shutil.copy2(src, dst)
        elif self.mode == "hardlink":
            os.link(src, dst)
        else:  # symlink
            os.symlink(src, dst)

    def _target_dirs(self, split: str) -> tuple[Path, Path]:
        if split == self.train_split:
            return self.img_train, self.lab_train
        if split == self.val_split:
            return self.img_val, self.lab_val
        # ignora outros splits
        return Path(), Path()

    def build(self) -> Path:
        self._ensure_dirs()
        rows = self._read_manifest()

        n_train = 0
        n_val = 0

        for r in rows:
            img_dir, lab_dir = self._target_dirs(r.split)
            if not img_dir:
                continue

            img_src = Path(r.image_path)
            if not img_src.exists():
                continue

            img_dst = img_dir / img_src.name
            self._place(img_src, img_dst)

            if r.label_path:
                lab_src = Path(r.label_path)
                if lab_src.exists():
                    lab_dst = lab_dir / f"{img_src.stem}.txt"
                    self._place(lab_src, lab_dst)

            if img_dir == self.img_train:
                n_train += 1
            else:
                n_val += 1

        yaml_path = self.write_data_yaml()

        print(f"[OK] Dataset Ultralytics em: {self.out_dir}")
        print(f"[OK] train: {n_train} | val: {n_val}")
        print(f"[OK] data.yaml: {yaml_path}")
        return yaml_path

    def write_data_yaml(self) -> Path:
        names = self._read_classes()
        yaml_path = self.out_dir / "data.yaml"

        lines = []
        lines.append(f"path: {self.out_dir.resolve().as_posix()}")
        lines.append("train: images/train")
        lines.append("val: images/val")
        lines.append("names:")
        for k, v in names.items():
            # escape simples
            v2 = v.replace('"', '\\"')
            lines.append(f"  {k}: \"{v2}\"")

        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path
