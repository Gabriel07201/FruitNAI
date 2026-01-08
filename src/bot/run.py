import math
import threading
import time
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

# Configuração - tuning)
# Detecção/tempo
MAX_DET_AGE_S = 0.03          # idade máxima da detecção (evita cortar no passado)
MIN_ACTION_INTERVAL_S = 0.06  # intervalo mínimo entre cortes
FOCUS_EVERY_S = 2.50          # refoca a janela a cada N segundos

MIN_FRUIT_CONF = 0.35         # confiança mínima para considerar fruta
MIN_FRUIT_AREA = 800          # área mínima (px^2) para filtrar frutas pequenas
RECENT_TTL_S = 0.16           # bloqueia repetir corte no mesmo lugar por N segundos
RECENT_RADIUS_PX = 85         # raio de bloqueio para cortes recentes

# Predictor
PREDICT_IMGSZ = 640           # tamanho da imagem para o modelo
PREDICT_IOU_THRES = 0.45      # IoU para NMS

# Parâmetros da ação
SINGLE_FAST_PARAMS = {        # slice rápido quando velocidade estimada é alta
    "length": 120,
    "down_wait": 0.01,
    "duration": 0.05,
    "steps": 2,
}
SINGLE_UNKNOWN_PARAMS = {     # slice mais generoso quando velocidade é desconhecida
    "length": 150,
    "down_wait": 0.02,
    "duration": 0.07,
    "steps": 2,
}
PAIR_PARAMS = {               # params de slice para pares (length é reservado)
    "length": 150,
    "down_wait": 0.02,
    "duration": 0.07,
    "steps": 3,
}
PAIR_SLICE_PARAMS = {k: v for k, v in PAIR_PARAMS.items() if k != "length"}
PRED_DURATION_FRAC = 0.7       # fração da duração usada para prever posição

# Velocidade/seleção
SPEED_FAST_THRESHOLD = 40.0    # troca para slice rápido acima desse valor
SPEED_DIR_THRESHOLD = 60.0     # usa direção da velocidade acima desse valor
TOP_FRUITS_LIMIT = 6           # máximo de frutas consideradas para pares
PAIR_MIN_DIST_PX = 55          # distância mínima entre frutas do par
PAIR_MAX_DIST_PX = 320         # distância máxima entre frutas do par

# Scoring do par
PAIR_SCORE_Y_WEIGHT = 1.0
PAIR_SCORE_CONF_WEIGHT = 260.0
PAIR_SCORE_DIST_WEIGHT = 0.10
PAIR_SCORE_MIN = -1e18

# Segurança/margens
WINDOW_MARGIN_PX = 14          # margem para clamp/slice dentro da janela
FOCUS_POINT_X = 90             # ponto de foco dentro da janela
FOCUS_POINT_Y = 90
SEGMENT_OFFSET_PX = 70         # offset do segmento para recheck de bomba
OVERSHOOT_BASE_PX = 20         # overshoot base do slice
OVERSHOOT_DIAG_FACTOR = 0.25   # fator sobre a diagonal do bbox
BOMB_SAFE_BASE_PX = 45         # raio base de segurança contra bombas
BOMB_SAFE_DIAG_FACTOR = 0.35   # fator sobre a diagonal do bbox da bomba
INSTANT_SAFE_BASE_PX = 50      # raio base no recheck instantâneo
BOMB_PRED_SAFE_BASE_PX = 50    # raio base na previsão de bomba
BOMB_PRED_SAFE_DIAG_FACTOR = 0.38  # fator sobre a diagonal na previsão

# Limites/epsilons
TIME_EPS_S = 1e-6              # evita divisões por zero
SEGMENT_EPS = 1e-9             # epsilon para degenerate segment
VELOCITY_MATCH_MAX_DIST_PX = 140  # distância máxima para match de velocidade

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
PERP_ANGLE_OFFSET_DEG = 90.0

# Controller
CONTROLLER_PAUSE_S = 0.0       # pausa interna do controlador


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


def _segment_is_bomb_safe(ax, ay, bx, by, bombs, *, base_safe_px: int = BOMB_PRED_SAFE_BASE_PX) -> bool:
    for b in bombs:
        cx, cy = _det_center(b)
        bw, bh = _det_wh(b)
        safe = max(base_safe_px, int(BOMB_SAFE_DIAG_FACTOR * math.hypot(bw, bh)))
        dist = _dist_point_to_segment(cx, cy, ax, ay, bx, by)
        if dist <= safe:
            return False
    return True


def _nearest_match_velocity(cur_pts, prev_pts, dt, *, max_dist=VELOCITY_MATCH_MAX_DIST_PX):
    if dt <= TIME_EPS_S or not prev_pts:
        return [(0.0, 0.0) for _ in cur_pts]

    vels = []
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
        else:
            px, py = best
            vels.append(((cx - px) / dt, (cy - py) / dt))
    return vels


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

    controller = Controller(pause=CONTROLLER_PAUSE_S, fail_safe=True)

    # Shared state
    lock = threading.Lock()
    latest_dets = []
    latest_region = None
    latest_ts = 0.0        # timestamp do FRAME (não do fim da inferência)
    latest_seq = 0
    
    
    predictor = YoloOnnxPredictor(
        onnx_path=onnx_path,
        imgsz=PREDICT_IMGSZ,
        conf_thres=MIN_FRUIT_CONF,
        iou_thres=PREDICT_IOU_THRES,
        class_agnostic=False,
    )


    # Estado interno do worker (pra estimar velocidade)
    prev_fruit_pts = []
    prev_bomb_pts = []
    prev_ts = None

    def action_worker():
        nonlocal prev_fruit_pts, prev_bomb_pts, prev_ts

        last_seen_seq = -1
        last_focus_t = 0.0
        last_action_t = 0.0

        recent_targets = []  # (x, y, t)

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

        def predict_point(x, y, vx, vy, tsec):
            return int(x + vx * tsec), int(y + vy * tsec)

        def last_instant_bomb_check(ax, ay, bx, by) -> bool:
            with lock:
                dets2 = list(latest_dets)
            bombs2 = [d for d in dets2 if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]
            if not bombs2:
                return True
            return _segment_is_bomb_safe(ax, ay, bx, by, bombs2, base_safe_px=INSTANT_SAFE_BASE_PX)

        def clamp_inside_window(x, y, w, h, margin):
            return (margin <= x <= (w - margin)) and (margin <= y <= (h - margin))

        while not state.shutdown.is_set():
            if not state.running.is_set():
                time.sleep(SLEEP_WHEN_STOPPED_S)
                continue

            with lock:
                dets = list(latest_dets)
                region = latest_region
                ts = latest_ts
                seq = latest_seq

            if region is None or seq == last_seen_seq:
                time.sleep(SLEEP_NO_NEW_FRAME_S)
                continue
            last_seen_seq = seq

            now = time.time()
            age = now - ts  # ts é do frame
            if age > MAX_DET_AGE_S:
                # det velha -> não age (evita cortar "no passado")
                time.sleep(SLEEP_OLD_DET_S)
                continue

            # cooldown global
            if (now - last_action_t) < MIN_ACTION_INTERVAL_S:
                time.sleep(SLEEP_ACTION_COOLDOWN_S)
                continue

            fruits_raw = [d for d in dets if int(getattr(d, "cls", -1)) == FRUIT_CLASS_ID]
            bombs = [d for d in dets if int(getattr(d, "cls", -1)) == BOMB_CLASS_ID]
            if not fruits_raw:
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
                continue

            # Estima velocidades
            cur_fruit_pts = [_det_center(d) for d in fruits]
            cur_bomb_pts = [_det_center(d) for d in bombs]

            if prev_ts is None:
                dtv = 0.0
            else:
                dtv = max(TIME_EPS_S, ts - prev_ts)

            fruit_vels = _nearest_match_velocity(cur_fruit_pts, prev_fruit_pts, dtv)
            bomb_vels = _nearest_match_velocity(cur_bomb_pts, prev_bomb_pts, dtv)

            prev_fruit_pts = cur_fruit_pts
            prev_bomb_pts = cur_bomb_pts
            prev_ts = ts

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

            # Ordena frutas por: mais embaixo + maior conf
            fruits_scored = []
            for i, d in enumerate(fruits):
                cx, cy = cur_fruit_pts[i]
                vx, vy = fruit_vels[i]
                conf = float(getattr(d, "conf", 0.0))
                fruits_scored.append((cy, conf, i, d, cx, cy, vx, vy))
            fruits_scored.sort(key=lambda t: (-t[0], -t[1]))

            # -------------------------
            # Tenta par (2 frutas)
            # -------------------------
            best_pair = None
            best_score = PAIR_SCORE_MIN
            top = fruits_scored[:TOP_FRUITS_LIMIT]

            if len(top) >= 2:
                for a in range(len(top)):
                    for b in range(a + 1, len(top)):
                        _, confa, ia, da, cxa, cya, vxa, vya = top[a]
                        _, confb, ib, db, cxb, cyb, vxb, vyb = top[b]

                        dist_ab = math.hypot(cxb - cxa, cyb - cya)
                        if dist_ab < PAIR_MIN_DIST_PX or dist_ab > PAIR_MAX_DIST_PX:
                            continue

                        t_pred = age + PAIR_PARAMS["down_wait"] + PAIR_PARAMS["duration"] * PRED_DURATION_FRAC
                        ax, ay = predict_point(cxa, cya, vxa, vya, t_pred)
                        bx, by = predict_point(cxb, cyb, vxb, vyb, t_pred)

                        if not clamp_inside_window(ax, ay, region.width, region.height, WINDOW_MARGIN_PX):
                            continue
                        if not clamp_inside_window(bx, by, region.width, region.height, WINDOW_MARGIN_PX):
                            continue

                        midx = int((ax + bx) / 2)
                        midy = int((ay + by) / 2)
                        if is_recent(midx, midy, now):
                            continue

                        # checa bombas previstas
                        safe = True
                        for k, bd in enumerate(bombs):
                            bcx, bcy = cur_bomb_pts[k]
                            bvx, bvy = bomb_vels[k]
                            px, py = predict_point(bcx, bcy, bvx, bvy, t_pred)
                            bw, bh = _det_wh(bd)
                            safe_r = max(BOMB_PRED_SAFE_BASE_PX, int(BOMB_PRED_SAFE_DIAG_FACTOR * math.hypot(bw, bh)))
                            if _dist_point_to_segment(px, py, ax, ay, bx, by) <= safe_r:
                                safe = False
                                break
                        if not safe:
                            continue

                        avg_y = 0.5 * (ay + by)
                        cmin = min(confa, confb)
                        score = (avg_y * PAIR_SCORE_Y_WEIGHT) + (cmin * PAIR_SCORE_CONF_WEIGHT) - (dist_ab * PAIR_SCORE_DIST_WEIGHT)

                        if score > best_score:
                            wa, ha = _det_wh(da)
                            wb, hb = _det_wh(db)
                            avg_diag = 0.5 * (math.hypot(wa, ha) + math.hypot(wb, hb))
                            dyn_overshoot = max(OVERSHOOT_BASE_PX, int(OVERSHOOT_DIAG_FACTOR * avg_diag))
                            best_score = score
                            best_pair = (ax, ay, bx, by, midx, midy, dyn_overshoot)

            try:
                if best_pair is not None:
                    ax, ay, bx, by, midx, midy, dyn_overshoot = best_pair

                    if not last_instant_bomb_check(ax, ay, bx, by):
                        continue

                    controller.slice_segment_in_window(
                        ax, ay, bx, by,
                        ScreenOffset(region.left, region.top),
                        window_w=region.width,
                        window_h=region.height,
                        margin=WINDOW_MARGIN_PX,
                        overshoot=dyn_overshoot,
                        **PAIR_SLICE_PARAMS
                    )

                    last_action_t = time.time()
                    add_recent(midx, midy, last_action_t)
                    continue

                # -------------------------
                # Fallback: 1 fruta
                # -------------------------
                _, conf0, i0, d0, cx, cy, vx, vy = fruits_scored[0]

                speed = math.hypot(vx, vy)

                # latência esperada até metade do swipe
                # (age já inclui inferência, porque ts é do frame)
                base_params = SINGLE_FAST_PARAMS if speed > SPEED_FAST_THRESHOLD else SINGLE_UNKNOWN_PARAMS
                w0, h0 = _det_wh(d0)
                dyn_overshoot = max(OVERSHOOT_BASE_PX, int(OVERSHOOT_DIAG_FACTOR * math.hypot(w0, h0)))
                t_pred = age + base_params["down_wait"] + base_params["duration"] * PRED_DURATION_FRAC
                px, py = predict_point(cx, cy, vx, vy, t_pred)

                if not clamp_inside_window(px, py, region.width, region.height, WINDOW_MARGIN_PX):
                    continue

                if is_recent(px, py, now):
                    continue

                # Direção: perpendicular à velocidade quando disponível
                if speed > SPEED_DIR_THRESHOLD:
                    ang = math.degrees(math.atan2(vy, vx)) + PERP_ANGLE_OFFSET_DEG
                else:
                    ang = DEFAULT_SLICE_ANGLE_DEG

                # Segmento pra recheck de bomba no último instante
                ax = px - SEGMENT_OFFSET_PX
                ay = py + SEGMENT_OFFSET_PX
                bx = px + SEGMENT_OFFSET_PX
                by = py - SEGMENT_OFFSET_PX
                if not last_instant_bomb_check(ax, ay, bx, by):
                    continue

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

                last_action_t = time.time()
                add_recent(px, py, last_action_t)

            except pyautogui.FailSafeException:
                state.shutdown.set()
                break
            except Exception as e:
                print("Erro no action_worker:", repr(e))
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

            # Publica para o worker antes de draw/imshow
            with lock:
                latest_dets = dets
                latest_region = region
                latest_ts = t_frame     # <<<< era t1; agora é o frame
                latest_seq += 1

            # Debug
            vis = predictor.draw(frame, dets)

            now = time.time()
            dt = max(TIME_EPS_S, now - last_t)
            last_t = now
            fps = FPS_SMOOTHING_OLD * fps + FPS_SMOOTHING_NEW * (1.0 / dt)

            status = "RUN" if state.running.is_set() else "STOP"
            infer_ms = (t1 - t0) * 1000.0

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

            k = cv2.waitKey(DEBUG_WAITKEY_MS) & 0xFF
            if k == ord("q"):
                state.shutdown.set()
                break

    finally:
        listener.stop()
        cv2.destroyAllWindows()


def main():
    state = BotState()
    bot_loop(state)


if __name__ == "__main__":
    main()
