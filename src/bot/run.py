import math
import threading
import time
import json
import os
from dataclasses import dataclass

import cv2
import pyautogui
from pynput import keyboard

from .state import BotState
from .capture import set_dpi_awareness, GameCapture, list_visible_window_titles
from .predictor import YoloOnnxPredictor
from .controller import Controller, ScreenOffset

TITLE_SUBSTRING = "Fruit Ninja"
ONNX_PATH = "models/runs/fruitninja_yolo11n/weights/best.onnx"

FRUIT_CLASS_ID = 0
BOMB_CLASS_ID = 1

# TUNING PRINCIPAL
# Limita cortes com detecção antiga (evita agir em frames atrasados).
MAX_DET_AGE_S = 0.028               # <<<<<< (antes 0.02)
# Evita spam de cortes impondo um intervalo mínimo entre ações.
MIN_ACTION_INTERVAL_S = 0.06
# Refoca a janela do jogo periodicamente para manter o foco ativo.
FOCUS_EVERY_S = 1.50

# Confiança mínima para reduzir falsos positivos de frutas.
MIN_FRUIT_CONF = 0.6
# Área mínima para ignorar frutas pequenas demais.
MIN_FRUIT_AREA = 600
# Tempo de bloqueio para não cortar repetidamente o mesmo local.
RECENT_TTL_S = 0.03
# IOU mínimo para considerar que é a mesma fruta recentemente cortada.
RECENT_IOU_THR = 0.12
# Distância máxima entre centros para considerar repetição de corte.
RECENT_CENTER_PX = 85
# Multiplicador de área para tornar o bloqueio mais conservador.
RECENT_AREA_BOOST = 1.35
# IOU mais permissivo quando a área cresceu (ex.: fruta se aproximando).
RECENT_AREA_IOU_THR = 0.20
# Distância maior para bloquear frutas com área aumentada.
RECENT_AREA_CENTER_PX = 110

# Limites geométricos (distâncias, margens e raios de segurança)
# distância mínima entre frutas para considerar par
MIN_PAIR_DISTANCE_PX = 90
# distância máxima entre frutas para considerar par
MAX_PAIR_DISTANCE_PX = 230
# margem para manter pontos dentro da janela
WINDOW_MARGIN_PX = 14

# offset do segmento de recheck de bomba
INSTANT_BOMB_SEGMENT_OFFSET_PX = 70
# overshoot mínimo em pixels
OVERSHOOT_MIN_PX = 42
# fator do overshoot baseado na diagonal da fruta
OVERSHOOT_DIAG_FACTOR = 0.15

# Pesos de scoring e thresholds de decisão
# peso do avg_y no score
SCORE_WEIGHT_AVG_Y = 1.0
# peso da confiança mínima no score
SCORE_WEIGHT_CONF = 260.0
# penalidade por distância no score
SCORE_DISTANCE_PENALTY = 0.10
# número de frutas consideradas no pairing
TOP_FRUITS_FOR_PAIR = 1

# velocidade mínima para usar predição rápida
SINGLE_SPEED_FAST_THRESHOLD = 9999
SINGLE_SPEED_ANG_THRESHOLD = 9999

# fator de duração usado na predição
PREDICT_DURATION_FACTOR = 0.32

# posição do foco na janela
FOCUS_WINDOW_X = 90
FOCUS_WINDOW_Y = 90

# segurança de corte em pixels para bombas
SAFE_BASE_PX = 90
SAFE_PREDICT_BASE_PX = 85
SAFE_BOMB_DIAG_FACTOR = 0.6
SAFE_PREDICT_DIAG_FACTOR = 0.7

# compensação + anti “cortes demais”
ACTION_OVERHEAD_S = 0.013           # <<<<<< medido no log (~13ms)
POST_HIT_PAUSE_S = 0.10             # pausa curta após hit confirmado
OUTCOME_WINDOW_S = 0.18             # janela pra decidir hit/miss
OUTCOME_IOU_THR = 0.10
OUTCOME_CENTER_PX = 90
OUTCOME_INTACT_IOU_THR = 0.45        # IoU alto => provável intacto
OUTCOME_INTACT_AREA_RATIO = 0.75     # área >= 75% da caixa-alvo => provável intacto
OUTCOME_SPLIT_AREA_RATIO = 0.60      # área <= 60% => forte indício de pedaço
OUTCOME_SPLIT_MIN_NEAR = 2           # 2+ dets perto => split (duas partes)


# velocidades
VEL_MATCH_MAX_DIST = 45             # px
VEL_MAX_SPEED = 900                 # px/s (clamp)
VEL_MIN_SPEED_USE = 260             # px/s (abaixo disso zera a previsão)

# Debug
DEBUG_DIR = "logs"
DEBUG_SKIP_AGE_EVERY = 25           # rate limit para skip-age


class DebugLogger:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def log(self, kind: str, **fields):
        obj = {"kind": kind, "ts": time.time()}
        obj.update(fields)
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def _det_center(d):
    return int((d.x1 + d.x2) / 2), int((d.y1 + d.y2) / 2)


def _det_wh(d):
    return int(abs(d.x2 - d.x1)), int(abs(d.y2 - d.y1))


def _det_xyxy(d):
    x1 = int(min(d.x1, d.x2))
    y1 = int(min(d.y1, d.y2))
    x2 = int(max(d.x1, d.x2))
    y2 = int(max(d.y1, d.y2))
    return x1, y1, x2, y2


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _dist_point_to_segment(px, py, ax, ay, bx, by) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-9:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _segment_is_bomb_safe(ax, ay, bx, by, bombs, *, base_safe_px: int = SAFE_BASE_PX) -> bool:
    for b in bombs:
        cx, cy = _det_center(b)
        bw, bh = _det_wh(b)
        safe = max(base_safe_px, int(SAFE_BOMB_DIAG_FACTOR * math.hypot(bw, bh)))
        dist = _dist_point_to_segment(cx, cy, ax, ay, bx, by)
        if dist <= safe:
            return False
    return True


def _nearest_match_velocity_with_quality(cur_pts, prev_pts, dt, *, max_dist=140):
    """
    Retorna (vels, match_dists) alinhados a cur_pts.
    match_dists[i] = distância px do melhor match.
    """
    if dt <= 1e-6 or not prev_pts:
        return [(0.0, 0.0) for _ in cur_pts], [None for _ in cur_pts]

    vels = []
    dists = []
    for (cx, cy) in cur_pts:
        best = None
        best_d2 = float("inf")
        for (px, py) in prev_pts:
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = (px, py)

        if best is None or best_d2 > (max_dist * max_dist):
            vels.append((0.0, 0.0))
            dists.append(None)
            continue

        px, py = best
        dist = math.sqrt(best_d2)

        vx = (cx - px) / dt
        vy = (cy - py) / dt

        # gating por qualidade do match (distância)
        if dist > VEL_MATCH_MAX_DIST:
            vx, vy = 0.0, 0.0
        else:
            # clamp velocidade máxima
            sp = math.hypot(vx, vy)
            if sp > VEL_MAX_SPEED:
                s = VEL_MAX_SPEED / max(1e-9, sp)
                vx *= s
                vy *= s
                sp = VEL_MAX_SPEED

            # NOVO: velocidade condicional (desliga previsão se muito baixa)
            if sp < VEL_MIN_SPEED_USE:
                vx, vy = 0.0, 0.0

        vels.append((vx, vy))
        dists.append(dist)

    return vels, dists



def _shift_box_to_pred(d, px, py):
    cx, cy = _det_center(d)
    dx = px - cx
    dy = py - cy
    x1, y1, x2, y2 = _det_xyxy(d)
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _toggle_running(state: BotState):
    if state.running.is_set():
        state.running.clear()
        print("[F2] Bot: PARADO")
    else:
        state.running.set()
        print("[F2] Bot: RODANDO")


def bot_loop(
    state: BotState,
    title_substring: str = TITLE_SUBSTRING,
    onnx_path: str = ONNX_PATH,
):
    set_dpi_awareness()

    def on_press(key):
        if key == keyboard.Key.f2:
            _toggle_running(state)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Janelas visíveis (amostra):")
    for t in list_visible_window_titles():
        print(" -", t)

    capture = GameCapture(title_substring=title_substring, bring_foreground=True)
    controller = Controller(pause=0.0, fail_safe=True)

    predictor = YoloOnnxPredictor(
        onnx_path=onnx_path,
        imgsz=640,
        conf_thres=MIN_FRUIT_CONF,
        iou_thres=0.45,
        class_agnostic=False,
    )

    # Log file
    ts_name = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(DEBUG_DIR, f"bot_debug_{ts_name}.jsonl")
    dbg = DebugLogger(log_path)

    # log config
    dbg.log(
    "config",
    # principais
    MAX_DET_AGE_S=MAX_DET_AGE_S,
    MIN_ACTION_INTERVAL_S=MIN_ACTION_INTERVAL_S,
    FOCUS_EVERY_S=FOCUS_EVERY_S,
    MIN_FRUIT_CONF=MIN_FRUIT_CONF,
    MIN_FRUIT_AREA=MIN_FRUIT_AREA,
    # recent
    RECENT_TTL_S=RECENT_TTL_S,
    RECENT_IOU_THR=RECENT_IOU_THR,
    RECENT_CENTER_PX=RECENT_CENTER_PX,
    RECENT_AREA_BOOST=RECENT_AREA_BOOST,
    RECENT_AREA_IOU_THR=RECENT_AREA_IOU_THR,
    RECENT_AREA_CENTER_PX=RECENT_AREA_CENTER_PX,
    # pair
    MIN_PAIR_DISTANCE_PX=MIN_PAIR_DISTANCE_PX,
    MAX_PAIR_DISTANCE_PX=MAX_PAIR_DISTANCE_PX,
    WINDOW_MARGIN_PX=WINDOW_MARGIN_PX,
    TOP_FRUITS_FOR_PAIR=TOP_FRUITS_FOR_PAIR,
    SCORE_DISTANCE_PENALTY=SCORE_DISTANCE_PENALTY,
    # slice
    INSTANT_BOMB_SEGMENT_OFFSET_PX=INSTANT_BOMB_SEGMENT_OFFSET_PX,
    OVERSHOOT_MIN_PX=OVERSHOOT_MIN_PX,
    OVERSHOOT_DIAG_FACTOR=OVERSHOOT_DIAG_FACTOR,
    # predict/outcome
    ACTION_OVERHEAD_S=ACTION_OVERHEAD_S,
    POST_HIT_PAUSE_S=POST_HIT_PAUSE_S,
    OUTCOME_WINDOW_S=OUTCOME_WINDOW_S,
    OUTCOME_IOU_THR=OUTCOME_IOU_THR,
    OUTCOME_CENTER_PX=OUTCOME_CENTER_PX,
    
    PREDICT_DURATION_FACTOR=PREDICT_DURATION_FACTOR,
    
    # velocidades
    VEL_MATCH_MAX_DIST=VEL_MATCH_MAX_DIST,
    VEL_MAX_SPEED=VEL_MAX_SPEED,
    VEL_MIN_SPEED_USE=VEL_MIN_SPEED_USE,
)

    # Shared state
    lock = threading.Lock()
    latest_dets = []
    latest_region = None
    latest_ts = 0.0
    latest_seq = 0

    single_params_fast = dict(length=120, down_wait=0.008, duration=0.04, steps=2)
    single_params_unknown = dict(length=150, down_wait=0.015, duration=0.055, steps=2)
    pair_params = dict(down_wait=0.015, duration=0.055, steps=3)

    prev_fruit_pts = []
    prev_bomb_pts = []
    prev_ts = None
    action_id_counter = 0

    def action_worker():
        nonlocal prev_fruit_pts, prev_bomb_pts, prev_ts, action_id_counter

        last_seen_seq = -1
        last_focus_t = 0.0
        last_action_t = 0.0
        last_hit_t = 0.0

        recent_boxes = []   # (x1,y1,x2,y2,t)
        recent_points = []  # (x,y,r,t)

        pending = None      # dict(action_id, kind, t, targets=[(box, center)])

        skip_age_count = 0

        def prune_recent(now):
            recent_boxes[:] = [(x1,y1,x2,y2,t) for (x1,y1,x2,y2,t) in recent_boxes if (now - t) <= RECENT_TTL_S]
            recent_points[:] = [(x,y,r,t) for (x,y,r,t) in recent_points if (now - t) <= RECENT_TTL_S]

        def add_recent_box(x1,y1,x2,y2, now):
            recent_boxes.append((x1,y1,x2,y2,now))

        def add_recent_point(x,y,r, now):
            recent_points.append((x,y,r,now))

        def is_recent_box(x1,y1,x2,y2, now, *, iou_thr, center_thr):
            prune_recent(now)
            box = (x1,y1,x2,y2)
            acx = 0.5*(x1+x2); acy = 0.5*(y1+y2)
            for rx1,ry1,rx2,ry2,rt in recent_boxes:
                ref = (rx1,ry1,rx2,ry2)
                if _box_iou(box, ref) >= iou_thr:
                    return True
                bcx = 0.5*(rx1+rx2); bcy = 0.5*(ry1+ry2)
                dx = acx-bcx; dy = acy-bcy
                if (dx*dx+dy*dy) <= (center_thr*center_thr):
                    return True
            return False

        def is_recent_point(x,y, now):
            prune_recent(now)
            for rx,ry,r,rt in recent_points:
                dx=x-rx; dy=y-ry
                if (dx*dx+dy*dy) <= (r*r):
                    return True
            return False

        def predict_point(x, y, vx, vy, tsec):
            return int(x + vx * tsec), int(y + vy * tsec)

        def last_instant_bomb_check(ax, ay, bx, by) -> bool:
            with lock:
                dets2 = list(latest_dets)
            bombs2 = [d for d in dets2 if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]
            if not bombs2:
                return True
            return _segment_is_bomb_safe(ax, ay, bx, by, bombs2, base_safe_px=SAFE_BASE_PX)

        def clamp_inside_window(x, y, w, h, margin):
            return (margin <= x <= (w - margin)) and (margin <= y <= (h - margin))

        def _box_area(b):
            x1, y1, x2, y2 = b
            return max(0, x2 - x1) * max(0, y2 - y1)

        def outcome_check(now, fruits_raw):
            """
            Decide hit/miss para o pending.

            Meta:
            - MISS (hit=False): fruta intacta ainda "no alvo"
            - HIT  (hit=True): fruta sumiu do alvo OU o que sobrou perto parece pedaço/split
            """
            nonlocal pending, last_hit_t
            if pending is None:
                return

            if (now - pending["t"]) > OUTCOME_WINDOW_S:
                dbg.log(
                    "outcome",
                    action_id=pending["action_id"],
                    action_kind=pending["kind"],
                    hit=None,
                    reason="timeout",
                )
                pending = None
                return

            thr2 = OUTCOME_CENTER_PX * OUTCOME_CENTER_PX

            intact_found = False
            split_found = False
            any_near = False

            # telemetria (ajuda a validar)
            best_iou_global = 0.0
            best_area_ratio_global = None
            near_count_global = 0

            for (tbox, (tx, ty)) in pending["targets"]:
                t_area = max(1, _box_area(tbox))

                best_iou = 0.0
                best_area_ratio = None
                near_count = 0

                for d in fruits_raw:
                    box = _det_xyxy(d)
                    cx, cy = _det_center(d)

                    iou = _box_iou(box, tbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_area_ratio = _box_area(box) / t_area

                    dx = cx - tx
                    dy = cy - ty
                    if (dx*dx + dy*dy) <= thr2:
                        near_count += 1

                # “perto” pode ser por IoU ou por centro (igual sua lógica original)
                is_near = (best_iou >= OUTCOME_IOU_THR) or (near_count > 0)
                if is_near:
                    any_near = True

                # atualiza telemetria global
                near_count_global += near_count
                if best_iou > best_iou_global:
                    best_iou_global = best_iou
                    best_area_ratio_global = best_area_ratio

                # classifica intacto vs split (somente se teve algo perto)
                if is_near and best_area_ratio is not None:
                    # intacto: IoU alto e área parecida com a caixa alvo
                    if (best_iou >= OUTCOME_INTACT_IOU_THR) and (best_area_ratio >= OUTCOME_INTACT_AREA_RATIO):
                        intact_found = True

                    # split: 2+ dets perto OU a melhor detecção é bem menor que a caixa alvo
                    if (near_count >= OUTCOME_SPLIT_MIN_NEAR) or (best_area_ratio <= OUTCOME_SPLIT_AREA_RATIO):
                        split_found = True

            # regra final
            if intact_found:
                hit = False
                reason = "intact"
            elif not any_near:
                hit = True
                reason = "gone"
            elif split_found:
                hit = True
                reason = "split"
            else:
                # caso raro: tem algo perto, mas não ficou claro
                hit = False
                reason = "ambiguous"

            dbg.log(
                "outcome",
                action_id=pending["action_id"],
                action_kind=pending["kind"],
                hit=hit,
                reason=reason,
                near_count=near_count_global,
                best_iou=best_iou_global,
                best_area_ratio=best_area_ratio_global,
                n_fruits_raw=len(fruits_raw),
            )

            if hit:
                last_hit_t = now
            pending = None


        while not state.shutdown.is_set():
            if not state.running.is_set():
                time.sleep(0.03)
                continue

            with lock:
                dets = list(latest_dets)
                region = latest_region
                ts = latest_ts
                seq = latest_seq

            if region is None or seq == last_seen_seq:
                time.sleep(0.003)
                continue
            last_seen_seq = seq

            now = time.time()
            age = now - ts
            if age > MAX_DET_AGE_S:
                skip_age_count += 1
                if (skip_age_count % DEBUG_SKIP_AGE_EVERY) == 0:
                    dbg.log("skip", seq=seq, reason="age", age_ms=age*1000.0, max_age_ms=MAX_DET_AGE_S*1000.0)
                continue

            if (now - last_hit_t) < POST_HIT_PAUSE_S:
                dbg.log("skip", seq=seq, reason="post_hit_pause", left_ms=(POST_HIT_PAUSE_S-(now-last_hit_t))*1000.0)
                continue

            if (now - last_action_t) < MIN_ACTION_INTERVAL_S:
                dbg.log("skip", seq=seq, reason="cooldown", left_ms=(MIN_ACTION_INTERVAL_S-(now-last_action_t))*1000.0)
                continue

            fruits_raw = [d for d in dets if int(getattr(d, "cls", -1)) == FRUIT_CLASS_ID]
            bombs = [d for d in dets if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]

            # tenta concluir outcome de ação anterior
            outcome_check(now, fruits_raw)

            if not fruits_raw:
                dbg.log("skip", seq=seq, reason="no_fruit_raw")
                continue

            prune_recent(now)

            # filtra frutas (conf/area)
            fruits = []
            for d in fruits_raw:
                conf = float(getattr(d, "conf", 0.0))
                w, h = _det_wh(d)
                area = w * h
                x1, y1, x2, y2 = _det_xyxy(d)

                boosted_min_area = MIN_FRUIT_AREA
                if is_recent_box(x1,y1,x2,y2, now, iou_thr=RECENT_IOU_THR, center_thr=RECENT_CENTER_PX):
                    boosted_min_area = int(MIN_FRUIT_AREA * RECENT_AREA_BOOST)

                if conf < MIN_FRUIT_CONF:
                    continue
                if area < boosted_min_area:
                    continue
                fruits.append(d)

            if not fruits:
                dbg.log("skip", seq=seq, reason="filtered_out", n_raw=len(fruits_raw))
                continue

            # estima velocidades
            cur_fruit_pts = [_det_center(d) for d in fruits]
            cur_bomb_pts = [_det_center(d) for d in bombs]

            dtv = 0.0 if prev_ts is None else max(1e-6, ts - prev_ts)

            fruit_vels, fruit_match_dists = _nearest_match_velocity_with_quality(cur_fruit_pts, prev_fruit_pts, dtv)
            bomb_vels, bomb_match_dists = _nearest_match_velocity_with_quality(cur_bomb_pts, prev_bomb_pts, dtv)

            prev_fruit_pts = cur_fruit_pts
            prev_bomb_pts = cur_bomb_pts
            prev_ts = ts

            # foco na janela
            if (now - last_focus_t) >= FOCUS_EVERY_S:
                controller.focus_window(
                    ScreenOffset(region.left, region.top),
                    window_w=region.width,
                    window_h=region.height,
                    margin=WINDOW_MARGIN_PX,
                    x=FOCUS_WINDOW_X,
                    y=FOCUS_WINDOW_Y,
                )
                last_focus_t = now

            # ordena frutas
            fruits_scored = []
            for i, d in enumerate(fruits):
                cx, cy = cur_fruit_pts[i]
                vx, vy = fruit_vels[i]
                conf = float(getattr(d, "conf", 0.0))
                md = fruit_match_dists[i]
                fruits_scored.append((cy, conf, i, d, cx, cy, vx, vy, md))
            fruits_scored.sort(key=lambda t: (-t[0], -t[1]))

            # tenta par
            best_pair = None
            best_score = -1e18
            top = fruits_scored[:TOP_FRUITS_FOR_PAIR]

            if len(top) >= 2:
                for a in range(len(top)):
                    for b in range(a + 1, len(top)):
                        _, confa, ia, da, cxa, cya, vxa, vya, mda = top[a]
                        _, confb, ib, db, cxb, cyb, vxb, vyb, mdb = top[b]

                        dist_ab = math.hypot(cxb - cxa, cyb - cya)
                        if dist_ab < MIN_PAIR_DISTANCE_PX or dist_ab > MAX_PAIR_DISTANCE_PX:
                            continue

                        t_pred = age + ACTION_OVERHEAD_S + pair_params["down_wait"] + pair_params["duration"] * PREDICT_DURATION_FACTOR
                        ax, ay = predict_point(cxa, cya, vxa, vya, t_pred)
                        bx, by = predict_point(cxb, cyb, vxb, vyb, t_pred)

                        if not clamp_inside_window(ax, ay, region.width, region.height, WINDOW_MARGIN_PX):
                            continue
                        if not clamp_inside_window(bx, by, region.width, region.height, WINDOW_MARGIN_PX):
                            continue

                        ax1, ay1, ax2, ay2 = _det_xyxy(da)
                        bx1, by1, bx2, by2 = _det_xyxy(db)
                        if is_recent_box(ax1,ay1,ax2,ay2, now, iou_thr=RECENT_IOU_THR, center_thr=RECENT_CENTER_PX):
                            continue
                        if is_recent_box(bx1,by1,bx2,by2, now, iou_thr=RECENT_IOU_THR, center_thr=RECENT_CENTER_PX):
                            continue

                        if is_recent_point(ax, ay, now) or is_recent_point(bx, by, now):
                            continue

                        # checa bombas previstas
                        safe = True
                        min_bomb_dist = None
                        for k, bd in enumerate(bombs):
                            bcx, bcy = cur_bomb_pts[k]
                            bvx, bvy = bomb_vels[k]
                            px, py = predict_point(bcx, bcy, bvx, bvy, t_pred)
                            bw, bh = _det_wh(bd)
                            safe_r = max(
                                SAFE_PREDICT_BASE_PX,
                                int(SAFE_PREDICT_DIAG_FACTOR * math.hypot(bw, bh)),
                            )
                            dd = _dist_point_to_segment(px, py, ax, ay, bx, by)
                            if min_bomb_dist is None or dd < min_bomb_dist:
                                min_bomb_dist = dd
                            if dd <= safe_r:
                                safe = False
                                break
                        if not safe:
                            continue

                        avg_y = 0.5 * (ay + by)
                        cmin = min(confa, confb)
                        score = (avg_y * SCORE_WEIGHT_AVG_Y) + (cmin * SCORE_WEIGHT_CONF) - (dist_ab * SCORE_DISTANCE_PENALTY)

                        if score > best_score:
                            wa, ha = _det_wh(da)
                            wb, hb = _det_wh(db)
                            avg_diag = 0.5 * (math.hypot(wa, ha) + math.hypot(wb, hb))
                            dyn_overshoot = max(OVERSHOOT_MIN_PX, int(OVERSHOOT_DIAG_FACTOR * avg_diag))
                            best_score = score
                            best_pair = (ax, ay, bx, by, dyn_overshoot, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, min_bomb_dist, t_pred, mda, mdb)

            try:
                if best_pair is not None:
                    ax, ay, bx, by, dyn_overshoot, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, min_bomb_dist, t_pred, mda, mdb = best_pair
                    if not last_instant_bomb_check(ax, ay, bx, by):
                        dbg.log("skip", seq=seq, reason="bomb_instant_pair")
                        continue

                    action_id_counter += 1
                    action_id = float(action_id_counter)

                    t0 = time.perf_counter()
                    controller.slice_segment_in_window(
                        ax, ay, bx, by,
                        ScreenOffset(region.left, region.top),
                        window_w=region.width,
                        window_h=region.height,
                        margin=WINDOW_MARGIN_PX,
                        overshoot=dyn_overshoot,
                        **pair_params
                    )
                    t1 = time.perf_counter()

                    planned_ms = (pair_params["down_wait"] + pair_params["duration"]) * 1000.0
                    exec_ms = (t1 - t0) * 1000.0

                    dbg.log(
                        "pair",
                        action_id=action_id,
                        seq=seq,
                        age_ms=age*1000.0,
                        t_pred=t_pred,
                        planned_ms=planned_ms,
                        action_exec_ms=exec_ms,
                        n_fruits=len(fruits),
                        n_bombs=len(bombs),
                        ax=ax, ay=ay, bx=bx, by=by,
                        overshoot=dyn_overshoot,
                        min_bomb_dist=min_bomb_dist,
                        match_dist_a=mda,
                        match_dist_b=mdb,
                    )

                    last_action_t = time.time()

                    # marca recentes (bbox deslocada pro ponto previsto)
                    ra1, ra2, ra3, ra4 = _shift_box_to_pred(da, ax, ay)
                    rb1, rb2, rb3, rb4 = _shift_box_to_pred(db, bx, by)
                    add_recent_box(ra1, ra2, ra3, ra4, last_action_t)
                    add_recent_box(rb1, rb2, rb3, rb4, last_action_t)

                    # marca recente por ponto
                    add_recent_point(ax, ay, RECENT_CENTER_PX, last_action_t)
                    add_recent_point(bx, by, RECENT_CENTER_PX, last_action_t)

                    # pendência pra outcome
                    pending = {
                        "action_id": action_id,
                        "kind": "pair",
                        "t": last_action_t,
                        "targets": [
                            ((ra1,ra2,ra3,ra4), (ax,ay)),
                            ((rb1,rb2,rb3,rb4), (bx,by)),
                        ],
                    }
                    continue

                _, conf0, i0, d0, cx, cy, vx, vy, md0 = fruits_scored[0]
                speed = math.hypot(vx, vy)

                base_params = single_params_fast if speed > SINGLE_SPEED_FAST_THRESHOLD else single_params_unknown
                w0, h0 = _det_wh(d0)
                dyn_overshoot = max(OVERSHOOT_MIN_PX, int(OVERSHOOT_DIAG_FACTOR * math.hypot(w0, h0)))

                t_pred = age + ACTION_OVERHEAD_S + base_params["down_wait"] + base_params["duration"] * PREDICT_DURATION_FACTOR
                px, py = predict_point(cx, cy, vx, vy, t_pred)

                if not clamp_inside_window(px, py, region.width, region.height, WINDOW_MARGIN_PX):
                    dbg.log("skip", seq=seq, reason="outside_single", px=px, py=py)
                    continue

                x1, y1, x2, y2 = _det_xyxy(d0)
                if is_recent_box(x1,y1,x2,y2, now, iou_thr=RECENT_IOU_THR, center_thr=RECENT_CENTER_PX):
                    dbg.log("skip", seq=seq, reason="recent_box_single")
                    continue
                if is_recent_point(px, py, now):
                    dbg.log("skip", seq=seq, reason="recent_point_single")
                    continue

                if speed > SINGLE_SPEED_ANG_THRESHOLD:
                    ang = math.degrees(math.atan2(vy, vx)) + 90.0
                else:
                    ang = -45.0

                ax = px - INSTANT_BOMB_SEGMENT_OFFSET_PX
                ay = py + INSTANT_BOMB_SEGMENT_OFFSET_PX
                bx = px + INSTANT_BOMB_SEGMENT_OFFSET_PX
                by = py - INSTANT_BOMB_SEGMENT_OFFSET_PX
                if not last_instant_bomb_check(ax, ay, bx, by):
                    dbg.log("skip", seq=seq, reason="bomb_instant_single")
                    continue

                action_id_counter += 1
                action_id = float(action_id_counter)

                t0 = time.perf_counter()
                controller.slice_in_window(
                    px, py,
                    ScreenOffset(region.left, region.top),
                    window_w=region.width,
                    window_h=region.height,
                    margin=WINDOW_MARGIN_PX,
                    angle_deg=ang,
                    overshoot=dyn_overshoot,
                    **base_params
                )
                t1 = time.perf_counter()

                planned_ms = (base_params["down_wait"] + base_params["duration"]) * 1000.0
                exec_ms = (t1 - t0) * 1000.0

                dbg.log(
                    "single",
                    action_id=action_id,
                    seq=seq,
                    age_ms=age*1000.0,
                    t_pred=t_pred,
                    planned_ms=planned_ms,
                    action_exec_ms=exec_ms,
                    n_fruits=len(fruits),
                    n_bombs=len(bombs),
                    px=px, py=py,
                    angle_deg=ang,
                    length=base_params["length"],
                    overshoot=dyn_overshoot,
                    speed=speed,
                    match_dist=md0,
                )

                last_action_t = time.time()

                rx1, ry1, rx2, ry2 = _shift_box_to_pred(d0, px, py)
                add_recent_box(rx1, ry1, rx2, ry2, last_action_t)
                add_recent_point(px, py, RECENT_CENTER_PX, last_action_t)

                pending = {
                    "action_id": action_id,
                    "kind": "single",
                    "t": last_action_t,
                    "targets": [((rx1,ry1,rx2,ry2), (px,py))],
                }

            except pyautogui.FailSafeException:
                state.shutdown.set()
                break
            except Exception as e:
                dbg.log("error", where="action_worker", err=repr(e))
                time.sleep(0.02)

        dbg.close()

    action_thread = threading.Thread(target=action_worker, daemon=True)
    action_thread.start()

    # Loop principal
    last_t = time.time()
    fps = 0.0

    try:
        while not state.shutdown.is_set():
            frame, region = capture.read()
            t_frame = time.time()

            t0 = time.time()
            dets = predictor.predict(frame)
            t1 = time.time()

            with lock:
                latest_dets = dets
                latest_region = region
                latest_ts = t_frame
                latest_seq += 1

            vis = predictor.draw(frame, dets)

            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            status = "RUN" if state.running.is_set() else "STOP"
            infer_ms = (t1 - t0) * 1000.0

            cv2.putText(
                vis,
                f"{region.width}x{region.height}  FPS:{fps:.1f}  dets:{len(dets)}  {status} (F2)  infer:{infer_ms:.0f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("bot_debug", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                state.shutdown.set()
                break

    finally:
        listener.stop()
        cv2.destroyAllWindows()
        dbg.close()


def main():
    state = BotState()
    bot_loop(state)


if __name__ == "__main__":
    main()
