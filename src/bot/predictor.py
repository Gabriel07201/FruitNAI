from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import onnxruntime as ort

ort.preload_dlls(directory="")

@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int


def _letterbox(
    img_bgr: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize com padding preservando aspect ratio.
    Retorna: img, r, (dw, dh)
    """
    h0, w0 = img_bgr.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w0, new_h / h0)
    if not scaleup:
        r = min(r, 1.0)

    w_unpad, h_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = new_w - w_unpad, new_h - h_unpad
    dw /= 2
    dh /= 2

    if (w0, h0) != (w_unpad, h_unpad):
        img = cv2.resize(img_bgr, (w_unpad, h_unpad), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """
    NMS simples em numpy.
    boxes: (N,4) em xyxy
    scores: (N,)
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)

    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_thres]

    return keep


class YoloOnnxPredictor:
    """
    Predictor genérico para YOLO exportado via Ultralytics para ONNX.
    Suporta saídas comuns:
      - (1, C, N) ou (1, N, C)
      - C pode ser 4+nc (estilo v8+) ou 4+1+nc (estilo v5)
    """
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        class_agnostic: bool = False,
    ):
        providers = []
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.class_agnostic = bool(class_agnostic)

        # Debug útil
        outs = self.sess.get_outputs()
        self.output_names = [o.name for o in outs]

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        img, r, (dw, dh) = _letterbox(frame_bgr, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # (3,H,W)
        x = np.expand_dims(x, axis=0)   # (1,3,H,W)
        return x, r, (dw, dh)

    def _standardize_output(self, out: np.ndarray) -> np.ndarray:
        """
        Retorna (N, C)
        """
        a = out
        # Alguns modelos retornam lista; aqui assumimos que já é ndarray
        if a.ndim == 4:
            # exemplos: (1,1,N,C) ou (1,N,1,C)
            a = np.squeeze(a)

        if a.ndim == 3:
            a = a[0]  # remove batch
            # caso (C,N) típico: C~84 e N~8400
            if a.shape[0] < a.shape[1]:
                a = a.T  # (N,C)
            # caso já (N,C): mantém
            return a

        if a.ndim == 2:
            return a

        raise RuntimeError(f"Formato de output inesperado: shape={out.shape}")

    def _decode_candidates(self, pred_nc: np.ndarray):
        """
        Gera duas interpretações:
          A) estilo v8+: [x,y,w,h, cls...]
          B) estilo v5:  [x,y,w,h, obj, cls...]
        Escolhe a que gera mais detecções acima do conf_thres.
        """
        # boxes (xywh)
        boxes_xywh = pred_nc[:, 0:4].astype(np.float32)

        # Converte para xyxy (ainda no espaço letterbox)
        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        boxes = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

        C = pred_nc.shape[1]

        # candidato A: sem obj explícito
        conf_a = None
        cls_a = None
        if C >= 5:
            scores_a = pred_nc[:, 4:].astype(np.float32)
            if scores_a.shape[1] > 0:
                cls_a = np.argmax(scores_a, axis=1).astype(np.int32)
                conf_a = np.max(scores_a, axis=1)

        # candidato B: com obj explícito
        conf_b = None
        cls_b = None
        if C >= 6:
            obj = pred_nc[:, 4].astype(np.float32)
            scores_b = pred_nc[:, 5:].astype(np.float32)
            if scores_b.shape[1] > 0:
                cls_b = np.argmax(scores_b, axis=1).astype(np.int32)
                conf_b = obj * np.max(scores_b, axis=1)
            else:
                # raro: só obj
                cls_b = np.zeros((pred_nc.shape[0],), dtype=np.int32)
                conf_b = obj

        # fallback se algo vier estranho
        if conf_a is None and conf_b is None:
            raise RuntimeError("Não consegui interpretar os scores do output (dimensões insuficientes).")

        # escolhe pelo “volume” de detecções válidas
        na = int(np.sum(conf_a >= self.conf_thres)) if conf_a is not None else -1
        nb = int(np.sum(conf_b >= self.conf_thres)) if conf_b is not None else -1

        if nb > na:
            return boxes, conf_b, cls_b
        return boxes, conf_a, cls_a

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        x, r, (dw, dh) = self._preprocess(frame_bgr)

        outs = self.sess.run(None, {self.input_name: x})
        out0 = outs[0]
        pred = self._standardize_output(out0)  # (N,C)

        boxes, conf, cls = self._decode_candidates(pred)
        if conf is None or cls is None:
            return []

        # filtra por confiança
        m = conf >= self.conf_thres
        boxes = boxes[m]
        conf = conf[m]
        cls = cls[m]

        if len(boxes) == 0:
            return []

        # desfaz letterbox para coordenadas da imagem original
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r

        boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

        # NMS
        dets: List[Detection] = []
        if self.class_agnostic:
            keep = _nms_xyxy(boxes, conf, self.iou_thres)
            for i in keep:
                dets.append(Detection(*boxes[i].tolist(), float(conf[i]), int(cls[i])))
            return dets

        # NMS por classe
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            keep = _nms_xyxy(boxes[idx], conf[idx], self.iou_thres)
            for k in keep:
                j = idx[k]
                dets.append(Detection(*boxes[j].tolist(), float(conf[j]), int(cls[j])))

        # ordena por confiança
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets

    @staticmethod
    def draw(frame_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        out = frame_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            color = (0, 0, 255) if d.cls == 1 else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"cls={d.cls} conf={d.conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return out
