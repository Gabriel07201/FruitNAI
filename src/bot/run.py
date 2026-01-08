import math
import os
import platform
import sys
import threading
import time
import concurrent.futures
import cv2
import pyautogui

from pynput import keyboard

from .state import BotState
from .capture import GameCapture, list_visible_window_titles
from .predictor import YoloOnnxPredictor
from .controller import Controller, ScreenOffset
from .logger import RunLogger


TITLE_SUBSTRING = "Fruit Ninja"
ONNX_PATH = "models/runs/fruitninja_yolo11n2/weights/best.onnx"

FRUIT_CLASS_ID = 0
BOMB_CLASS_ID = 1

# Configuração - tuning)
# Detecção/tempo
MAX_DET_AGE_S = 0.03          # idade máxima da detecção (evita cortar no passado)
MIN_ACTION_INTERVAL_S = 0.04  # intervalo mínimo entre cortes
FOCUS_EVERY_S = 5          # refoca a janela a cada N segundos

MIN_FRUIT_CONF = 0.82         # confiança mínima para considerar fruta
MIN_FRUIT_AREA = 2000          # área mínima (px^2) para filtrar frutas pequenas
RECENT_TTL_S = 0.25           # bloqueia repetir corte no mesmo lugar por N segundos
RECENT_RADIUS_PX = 90         # raio de bloqueio para cortes recentes

# Predictor
PREDICT_IMGSZ = 640           # tamanho da imagem para o modelo
PREDICT_IOU_THRES = 0.50      # IoU para NMS

# Parâmetros da ação
SINGLE_SHORT_PARAMS = {       # slice curto quando só há 1 fruta (do centro para fora)
    "length": 140,
    "down_wait": 0.01,
    "duration": 0.055,
    "steps": 2,
}

# Segurança/margens
WINDOW_MARGIN_PX = 14          # margem para clamp/slice dentro da janela
FOCUS_POINT_X = 90             # ponto de foco dentro da janela
FOCUS_POINT_Y = 90
SEGMENT_OFFSET_PX = 70         # offset do segmento para recheck de bomba
OVERSHOOT_BASE_PX = 20         # overshoot base do slice
OVERSHOOT_DIAG_FACTOR = 0.25   # fator sobre a diagonal do bbox
BOMB_SAFE_BASE_PX = 80         # raio base de segurança contra bombas
BOMB_SAFE_DIAG_FACTOR = 0.40   # fator sobre a diagonal do bbox da bomba
INSTANT_SAFE_BASE_PX = 80      # raio base no recheck instantâneo

# Limites/epsilons
TIME_EPS_S = 1e-6              # evita divisões por zero
SEGMENT_EPS = 1e-9             # epsilon para degenerate segment

# Esperas do loop
SLEEP_WHEN_STOPPED_S = 0.03
SLEEP_NO_NEW_FRAME_S = 0.003
SLEEP_OLD_DET_S = 0.002
SLEEP_ACTION_COOLDOWN_S = 0.001
SLEEP_WORKER_ERROR_S = 0.02

# Debug HUD
FPS_SMOOTHING_OLD = 0.9
FPS_SMOOTHING_NEW = 0.1
DEBUG_TEXT_POS = (10, 30)
DEBUG_TEXT_SCALE = 0.8
DEBUG_TEXT_THICKNESS = 2
DEBUG_TEXT_COLOR = (255, 255, 255)
DEBUG_WAITKEY_MS = 1

# Direção do slice
DEFAULT_SLICE_ANGLE_DEG = -45.0

# Controller
CONTROLLER_PAUSE_S = 0.0       # pausa interna do controlador

# Logger
FRAME_SAMPLE_EVERY = 15
SKIP_LOG_MIN_INTERVAL_S = 0.12
SKIP_AGG_EVERY_S = 5.0
POST_ACTION_DELAY_S = 0.16
POST_ACTION_NEAR_PX = 90


def _det_center(d):
    return int((d.x1 + d.x2) / 2), int((d.y1 + d.y2) / 2)


def _det_wh(d):
    return int(abs(d.x2 - d.x1)), int(abs(d.y2 - d.y1))


def _dist_point_to_segment(px, py, ax, ay, bx, by) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 <= SEGMENT_EPS:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _segment_is_bomb_safe(ax, ay, bx, by, bombs, *, base_safe_px: int = BOMB_SAFE_BASE_PX) -> bool:
    for b in bombs:
        cx, cy = _det_center(b)
        bw, bh = _det_wh(b)
        safe = max(base_safe_px, int(BOMB_SAFE_DIAG_FACTOR * math.hypot(bw, bh)))
        dist = _dist_point_to_segment(cx, cy, ax, ay, bx, by)
        if dist <= safe:
            return False
    return True


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
    start_ts = time.time()
    logger = RunLogger()
    logger.log_event(
        "run",
        {
            "pid": os.getpid(),
            "python": sys.version,
            "platform": platform.platform(),
            "title_substring": title_substring,
            "onnx_path": onnx_path,
            "start_ts_name": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": start_ts,
        },
    )

    def on_press(key):
        if key == keyboard.Key.f2:
            _toggle_running(state)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Janelas visíveis (amostra):")
    for t in list_visible_window_titles():
        print(" -", t)

    capture = GameCapture(title_substring=title_substring, bring_foreground=True)

    controller = Controller(pause=CONTROLLER_PAUSE_S, fail_safe=True)
    post_action_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Shared state
    lock = threading.Lock()
    latest_dets = []
    latest_region = None
    latest_ts = 0.0        # timestamp do FRAME (não do fim da inferência)
    latest_seq = 0
    latest_infer_ms = 0.0
    
    
    predictor = YoloOnnxPredictor(
        onnx_path=onnx_path,
        imgsz=PREDICT_IMGSZ,
        conf_thres=MIN_FRUIT_CONF,
        iou_thres=PREDICT_IOU_THRES,
        class_agnostic=False,
    )
    
    logger.log_event(
        "config",
        {
            "tuning_params": {
                "max_det_age_s": MAX_DET_AGE_S,
                "min_action_interval_s": MIN_ACTION_INTERVAL_S,
                "focus_every_s": FOCUS_EVERY_S,
                "min_fruit_conf": MIN_FRUIT_CONF,
                "min_fruit_area": MIN_FRUIT_AREA,
                "recent_ttl_s": RECENT_TTL_S,
                "recent_radius_px": RECENT_RADIUS_PX,
                "predict_imgsz": PREDICT_IMGSZ,
                "predict_iou_thres": PREDICT_IOU_THRES,
            },
            "slice_params": {
                "single_short_params": SINGLE_SHORT_PARAMS,
            },
            "safety_constants": {
                "window_margin_px": WINDOW_MARGIN_PX,
                "segment_offset_px": SEGMENT_OFFSET_PX,
                "overshoot_base_px": OVERSHOOT_BASE_PX,
                "overshoot_diag_factor": OVERSHOOT_DIAG_FACTOR,
                "bomb_safe_base_px": BOMB_SAFE_BASE_PX,
                "bomb_safe_diag_factor": BOMB_SAFE_DIAG_FACTOR,
                "instant_safe_base_px": INSTANT_SAFE_BASE_PX,
            },
            "focus_params": {
                "focus_point_x": FOCUS_POINT_X,
                "focus_point_y": FOCUS_POINT_Y,
            },
            "limits_eps": {
                "time_eps_s": TIME_EPS_S,
                "segment_eps": SEGMENT_EPS,
            },
            "loop_sleeps": {
                "sleep_when_stopped_s": SLEEP_WHEN_STOPPED_S,
                "sleep_no_new_frame_s": SLEEP_NO_NEW_FRAME_S,
                "sleep_old_det_s": SLEEP_OLD_DET_S,
                "sleep_action_cooldown_s": SLEEP_ACTION_COOLDOWN_S,
                "sleep_worker_error_s": SLEEP_WORKER_ERROR_S,
            },
            "debug_hud": {
                "fps_smoothing_old": FPS_SMOOTHING_OLD,
                "fps_smoothing_new": FPS_SMOOTHING_NEW,
                "debug_text_pos": DEBUG_TEXT_POS,
                "debug_text_scale": DEBUG_TEXT_SCALE,
                "debug_text_thickness": DEBUG_TEXT_THICKNESS,
                "debug_text_color": DEBUG_TEXT_COLOR,
                "debug_waitkey_ms": DEBUG_WAITKEY_MS,
            },
            "direction_params": {
                "default_slice_angle_deg": DEFAULT_SLICE_ANGLE_DEG,
            },
            "controller_params": {
                "controller_pause_s": CONTROLLER_PAUSE_S,
            },
            "logger_params": {
                "frame_sample_every": FRAME_SAMPLE_EVERY,
                "skip_log_min_interval_s": SKIP_LOG_MIN_INTERVAL_S,
                "skip_agg_every_s": SKIP_AGG_EVERY_S,
                "post_action_delay_s": POST_ACTION_DELAY_S,
                "post_action_near_px": POST_ACTION_NEAR_PX,
            },
            "class_ids": {
                "fruit_class_id": FRUIT_CLASS_ID,
                "bomb_class_id": BOMB_CLASS_ID,
            },
        },
    )


    def action_worker():

        last_seen_seq = -1
        last_focus_t = 0.0
        last_action_t = 0.0
        action_id = 0

        skip_counts = {}
        last_skip_log_t = 0.0
        last_skip_agg_t = time.time()

        recent_targets = []  # (x, y, t)
        prev_fruit_obs = []  # (cx, cy, ts_frame)

        def prune_recent(now):
            recent_targets[:] = [(x, y, t) for (x, y, t) in recent_targets if (now - t) <= RECENT_TTL_S]

        def is_recent(x, y, now):
            prune_recent(now)
            for rx, ry, rt in recent_targets:
                if (x - rx) * (x - rx) + (y - ry) * (y - ry) <= (RECENT_RADIUS_PX * RECENT_RADIUS_PX):
                    return True
            return False

        def add_recent(x, y, now):
            recent_targets.append((x, y, now))
            prune_recent(now)
        
        def log_skip(reason, seq, **fields):
            nonlocal last_skip_log_t, last_skip_agg_t
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            now = time.time()
            if (now - last_skip_log_t) >= SKIP_LOG_MIN_INTERVAL_S:
                payload = {"seq": seq, "reason": reason}
                payload.update(fields)
                logger.log_event("skip", payload)
                last_skip_log_t = now
            if (now - last_skip_agg_t) >= SKIP_AGG_EVERY_S:
                logger.log_event(
                    "skip_agg",
                    {
                        "seq": seq,
                        "interval_s": now - last_skip_agg_t,
                        "counts": dict(skip_counts),
                    },
                )
                skip_counts.clear()
                last_skip_agg_t = now

        def post_action_observe(seq, region, targets):
            time.sleep(POST_ACTION_DELAY_S)
            with lock:
                dets2 = list(latest_dets)
            fruits2 = [d for d in dets2 if int(getattr(d, "cls", -1)) == FRUIT_CLASS_ID]
            centers = [_det_center(d) for d in fruits2]
            min_dist = None
            near_count = 0
            for tx, ty in targets:
                for cx, cy in centers:
                    dist = math.hypot(cx - tx, cy - ty)
                    min_dist = dist if min_dist is None else min(min_dist, dist)
                    if dist <= POST_ACTION_NEAR_PX:
                        near_count += 1
            if min_dist is None:
                min_dist = -1.0
            if near_count == 0:
                result_hint = "gone"
            elif near_count >= 2:
                result_hint = "split_like"
            else:
                result_hint = "still_near"
            logger.log_event(
                "post_action",
                {
                    "seq": seq,
                    "near_count": near_count,
                    "min_center_dist": min_dist,
                    "result_hint": result_hint,
                    "region": {
                        "left": region.left,
                        "top": region.top,
                        "width": region.width,
                        "height": region.height,
                    },
                },
            )

        def last_instant_bomb_check(ax, ay, bx, by) -> bool:
            with lock:
                dets2 = list(latest_dets)
            bombs2 = [d for d in dets2 if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]
            if not bombs2:
                return True
            return _segment_is_bomb_safe(ax, ay, bx, by, bombs2, base_safe_px=INSTANT_SAFE_BASE_PX)

        def clamp_inside_window(x, y, w, h, margin):
            clamped_x = min(max(x, margin), w - margin)
            clamped_y = min(max(y, margin), h - margin)
            clamped = (clamped_x != x) or (clamped_y != y)
            return clamped_x, clamped_y, clamped

        while not state.shutdown.is_set():
            if not state.running.is_set():
                time.sleep(SLEEP_WHEN_STOPPED_S)
                continue

            with lock:
                dets = list(latest_dets)
                region = latest_region
                ts = latest_ts
                seq = latest_seq
                infer_ms = latest_infer_ms

            if region is None or seq == last_seen_seq:
                time.sleep(SLEEP_NO_NEW_FRAME_S)
                continue
            last_seen_seq = seq

            now = time.time()
            age = now - ts  # ts é do frame
            if age > MAX_DET_AGE_S:
                # det velha -> não age (evita cortar "no passado")
                log_skip(
                    "age",
                    seq,
                    age_ms=age * 1000.0,
                    max_age_ms=MAX_DET_AGE_S * 1000.0,
                )
                time.sleep(SLEEP_OLD_DET_S)
                continue

            # cooldown global
            if (now - last_action_t) < MIN_ACTION_INTERVAL_S:
                remaining = max(0.0, MIN_ACTION_INTERVAL_S - (now - last_action_t))
                log_skip(
                    "cooldown",
                    seq,
                    left_ms=remaining * 1000.0,
                    min_interval_ms=MIN_ACTION_INTERVAL_S * 1000.0,
                )
                time.sleep(SLEEP_ACTION_COOLDOWN_S)
                continue

            fruits_raw = [d for d in dets if int(getattr(d, "cls", -1)) == FRUIT_CLASS_ID]
            bombs = [d for d in dets if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]
            if not fruits_raw:
                log_skip("no_fruit_raw", seq, n_raw=0)
                continue

            # Filtra frutas (conf/area)
            fruits = []
            for d in fruits_raw:
                conf = float(getattr(d, "conf", 0.0))
                w, h = _det_wh(d)
                area = w * h
                if conf < MIN_FRUIT_CONF:
                    continue
                if area < MIN_FRUIT_AREA:
                    continue
                fruits.append(d)

            if not fruits:
                log_skip("filtered_out", seq, n_raw=len(fruits_raw))
                continue

            cur_fruit_pts = [_det_center(d) for d in fruits]

            # Foco (não toda hora)
            if (now - last_focus_t) >= FOCUS_EVERY_S:
                controller.focus_window(
                    ScreenOffset(region.left, region.top),
                    window_w=region.width,
                    window_h=region.height,
                    margin=WINDOW_MARGIN_PX,
                    x=FOCUS_POINT_X,
                    y=FOCUS_POINT_Y,
                )
                last_focus_t = now
                logger.log_event(
                    "focus",
                    {
                        "seq": seq,
                        "region": {
                            "left": region.left,
                            "top": region.top,
                            "width": region.width,
                            "height": region.height,
                        },
                        "x": FOCUS_POINT_X,
                        "y": FOCUS_POINT_Y,
                    },
                )

            # Ordena frutas por: mais embaixo + maior conf
            fruits_scored = []
            for i, d in enumerate(fruits):
                cx, cy = cur_fruit_pts[i]
                conf = float(getattr(d, "conf", 0.0))
                fruits_scored.append((cy, conf, i, d, cx, cy))
            fruits_scored.sort(key=lambda t: (-t[0], -t[1]))
            
            # Calcula predição de posição com base em velocidade simples
            def nearest_prev_point(cx, cy):
                if not prev_fruit_obs:
                    return None
                return min(
                    prev_fruit_obs,
                    key=lambda p: (cx - p[0]) * (cx - p[0]) + (cy - p[1]) * (cy - p[1]),
                )

            _, _, _, _, cx, cy = fruits_scored[0]
            prev_point = nearest_prev_point(cx, cy)
            vx = 0.0
            vy = 0.0
            if prev_point is not None:
                prev_cx, prev_cy, prev_ts = prev_point
                obs_dt = ts - prev_ts
                if obs_dt > TIME_EPS_S:
                    vx = (cx - prev_cx) / obs_dt
                    vy = (cy - prev_cy) / obs_dt

            action_delay_s = SINGLE_SHORT_PARAMS["down_wait"] + (SINGLE_SHORT_PARAMS["duration"] / 2.0)
            t_pred = now + (infer_ms / 1000.0) + age + action_delay_s
            dt_pred = t_pred - ts
            px = cx + vx * dt_pred
            py = cy + vy * dt_pred

            prev_fruit_obs = [(pt[0], pt[1], ts) for pt in cur_fruit_pts]
            
            target_current = (cx, cy)
            target_pred = (px, py)
            t_cross_s = action_delay_s
            infer_ms_last = infer_ms
            gesture = {
                "down_wait": SINGLE_SHORT_PARAMS["down_wait"],
                "duration": SINGLE_SHORT_PARAMS["duration"],
                "steps": SINGLE_SHORT_PARAMS["steps"],
                "length": SINGLE_SHORT_PARAMS["length"],
                "angle_deg": DEFAULT_SLICE_ANGLE_DEG,
            }
            clamped_x, clamped_y, clamped = clamp_inside_window(
                px,
                py,
                region.width,
                region.height,
                WINDOW_MARGIN_PX,
            )
            ax = px - SEGMENT_OFFSET_PX
            ay = py + SEGMENT_OFFSET_PX
            bx = px + SEGMENT_OFFSET_PX
            by = py - SEGMENT_OFFSET_PX
            segment_for_bomb = (ax, ay, bx, by)

            nearest_center = None
            dist_to_pred = None
            dist_to_current = None
            if cur_fruit_pts:
                nx, ny = min(
                    cur_fruit_pts,
                    key=lambda p: (p[0] - px) * (p[0] - px) + (p[1] - py) * (p[1] - py),
                )
                nearest_center = (nx, ny)
                dist_to_pred = math.hypot(nx - px, ny - py)
                dist_to_current = math.hypot(nx - cx, ny - cy)

            speed = math.hypot(vx, vy)
            if speed > 0.0:
                proj_along_v = ((px - cx) * vx + (py - cy) * vy) / speed
            else:
                proj_along_v = 0.0
            
            top_candidates = []
            for cy, conf, i, d, cx, cy2 in fruits_scored[:max(3, min(6, len(fruits_scored)))]:
                w, h = _det_wh(d)
                area = w * h
                top_candidates.append(
                    {
                        "cx": cx,
                        "cy": cy2,
                        "conf": conf,
                        "area": area,
                    }
                )

            logger.log_event(
                "decision",
                {
                    "seq": seq,
                    "age_ms": age * 1000.0,
                    "n_fruits_raw": len(fruits_raw),
                    "n_fruits_filtered": len(fruits),
                    "n_bombs": len(bombs),
                    "pred_dt_s": dt_pred,
                    "target_current": target_current,
                    "target_pred": target_pred,
                    "t_cross_s": t_cross_s,
                    "infer_ms_last": infer_ms_last,
                    "delay_s": action_delay_s,
                    "gesture": gesture,
                    "clamped": clamped,
                    "segment_for_bomb": segment_for_bomb,
                    "nearest_center": nearest_center,
                    "dist_to_pred": dist_to_pred,
                    "dist_to_current": dist_to_current,
                    "vx": vx,
                    "vy": vy,
                    "speed": speed,
                    "proj_along_v": proj_along_v,
                    "top_candidates": top_candidates,
                },
            )


            try:
                # -------------------------
                # Fallback: 1 fruta
                # -------------------------
                if clamped:
                    log_skip(
                        "outside",
                        seq,
                        target_pred=target_pred,
                        segment_for_bomb=segment_for_bomb,
                        clamped=clamped,
                        t_cross_s=t_cross_s,
                        infer_ms_last=infer_ms_last,
                    )
                    continue

                if is_recent(px, py, now):
                    log_skip(
                        "recent",
                        seq,
                        target_pred=target_pred,
                        segment_for_bomb=segment_for_bomb,
                        clamped=clamped,
                        t_cross_s=t_cross_s,
                        infer_ms_last=infer_ms_last,
                    )
                    continue

                ang = DEFAULT_SLICE_ANGLE_DEG

                # Segmento pra recheck de bomba no último instante
                if not last_instant_bomb_check(ax, ay, bx, by):
                    log_skip(
                        "bomb",
                        seq,
                        target_pred=target_pred,
                        segment_for_bomb=segment_for_bomb,
                        clamped=clamped,
                        t_cross_s=t_cross_s,
                        infer_ms_last=infer_ms_last,
                    )
                    continue

                action_id += 1
                logger.log_event(
                    "action",
                    {
                        "kind": "single_short",
                        "action_id": action_id,
                        "seq": seq,
                        "action_ts": time.time(),
                        "age_ms": age * 1000.0,
                        "target_current": target_current,
                        "target_pred": target_pred,
                        "t_cross_s": t_cross_s,
                        "infer_ms_last": infer_ms_last,
                        "delay_s": action_delay_s,
                        "gesture": {
                            **SINGLE_SHORT_PARAMS,
                            "angle_deg": ang,
                        },
                        "clamped": clamped,
                        "segment_for_bomb": segment_for_bomb,
                        "nearest_center": nearest_center,
                        "dist_to_pred": dist_to_pred,
                        "dist_to_current": dist_to_current,
                        "vx": vx,
                        "vy": vy,
                        "speed": speed,
                        "proj_along_v": proj_along_v,
                        "angle_deg": ang,
                        "params": SINGLE_SHORT_PARAMS,
                        "bomb_check": {
                            "instant_safe": True,
                            "bombs_n": len(bombs),
                        },
                        "recent": {
                            "recent_targets_count": len(recent_targets),
                            "recent_ttl_s": RECENT_TTL_S,
                            "recent_radius_px": RECENT_RADIUS_PX,
                        },
                    },
                )

                controller.slice_short_from_center_in_window(
                    px, py,
                    ScreenOffset(region.left, region.top),
                    window_w=region.width,
                    window_h=region.height,
                    margin=WINDOW_MARGIN_PX,
                    angle_deg=ang,
                    **SINGLE_SHORT_PARAMS
                )

                last_action_t = time.time()
                add_recent(px, py, last_action_t)
                post_action_executor.submit(post_action_observe, seq, region, [(px, py)])

            except pyautogui.FailSafeException:
                logger.log_event(
                    'shutdown',
                    {'reason': 'pyautogui_failsafe_triggered', 'seq': seq},
                )
                state.shutdown.set()
                break
            except Exception as e:
                print("Erro no action_worker:", repr(e))
                logger.log_event(
                    "error",
                    {"where": "action_worker", "err": repr(e), "seq": seq},
                )
                time.sleep(SLEEP_WORKER_ERROR_S)

    action_thread = threading.Thread(target=action_worker, daemon=True)
    action_thread.start()

    # Loop principal (captura + inferência + debug)
    last_t = time.time()
    fps = 0.0

    try:
        while not state.shutdown.is_set():
            frame, region = capture.read()
            t_frame = time.time()  # <<<< timestamp do frame

            t0 = time.time()
            dets = predictor.predict(frame)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000.0

            # Publica para o worker antes de draw/imshow
            with lock:
                latest_dets = dets
                latest_region = region
                latest_ts = t_frame     # <<<< era t1; agora é o frame
                latest_infer_ms = infer_ms
                latest_seq += 1
                seq = latest_seq

            # Debug
            vis = predictor.draw(frame, dets)

            now = time.time()
            dt = max(TIME_EPS_S, now - last_t)
            last_t = now
            fps = FPS_SMOOTHING_OLD * fps + FPS_SMOOTHING_NEW * (1.0 / dt)

            status = "RUN" if state.running.is_set() else "STOP"

            cv2.putText(
                vis,
                f"{region.width}x{region.height}  FPS:{fps:.1f}  dets:{len(dets)}  {status} (F2)  infer:{infer_ms:.0f}ms",
                DEBUG_TEXT_POS,
                cv2.FONT_HERSHEY_SIMPLEX,
                DEBUG_TEXT_SCALE,
                DEBUG_TEXT_COLOR,
                DEBUG_TEXT_THICKNESS,
                cv2.LINE_AA,
            )

            cv2.imshow("bot_debug", vis)
            
            if seq % FRAME_SAMPLE_EVERY == 0:
                fruits_raw = [d for d in dets if int(getattr(d, "cls", -1)) == FRUIT_CLASS_ID]
                bombs = [d for d in dets if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]
                logger.log_event(
                    "frame",
                    {
                        "seq": seq,
                        "ts_frame": t_frame,
                        "fps": fps,
                        "infer_ms": infer_ms,
                        "region": {
                            "left": region.left,
                            "top": region.top,
                            "width": region.width,
                            "height": region.height,
                        },
                        "dets_total": len(dets),
                        "fruit_raw": len(fruits_raw),
                        "bomb_raw": len(bombs),
                    },
                )

            k = cv2.waitKey(DEBUG_WAITKEY_MS) & 0xFF
            if k == ord("q"):
                state.shutdown.set()
                break

    finally:
        listener.stop()
        cv2.destroyAllWindows()
        logger.close()


def main():
    state = BotState()
    bot_loop(state)


if __name__ == "__main__":
    main()
