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

# Limita cortes com detecção antiga (evita agir em frames atrasados).
MAX_DET_AGE_S = 0.02
# Evita spam de cortes impondo um intervalo mínimo entre ações.
MIN_ACTION_INTERVAL_S = 0.06
# Refoca a janela do jogo periodicamente para manter o foco ativo.
FOCUS_EVERY_S = 2.50

# Confiança mínima para reduzir falsos positivos de frutas.
MIN_FRUIT_CONF = 0.51
# Área mínima para ignorar frutas pequenas demais.
MIN_FRUIT_AREA = 500
# Tempo de bloqueio para não cortar repetidamente o mesmo local.
RECENT_TTL_S = 0.2
# IOU mínimo para considerar que é a mesma fruta recentemente cortada.
RECENT_IOU_THR = 0.12
# Distância máxima entre centros para considerar repetição de corte.
RECENT_CENTER_PX = 85
# Multiplicador de área para tornar o bloqueio mais conservador.
RECENT_AREA_BOOST = 1.35
# IOU mais permissivo quando a área cresceu (ex.: fruta se aproximando).
RECENT_AREA_IOU_THR = 0.2
# Distância maior para bloquear frutas com área aumentada.
RECENT_AREA_CENTER_PX = 110

# Limites geométricos (distâncias, margens e raios de segurança)
MIN_PAIR_DISTANCE_PX = 55           # distância mínima entre frutas para considerar par
MAX_PAIR_DISTANCE_PX = 320          # distância máxima entre frutas para considerar par
WINDOW_MARGIN_PX = 14               # margem para manter pontos dentro da janela
INSTANT_BOMB_SEGMENT_OFFSET_PX = 70 # offset do segmento de recheck de bomba
OVERSHOOT_MIN_PX = 20               # overshoot mínimo em pixels
OVERSHOOT_DIAG_FACTOR = 0.25        # fator do overshoot baseado na diagonal da fruta

# Pesos de scoring e thresholds de decisão
SCORE_WEIGHT_AVG_Y = 1.0            # peso do avg_y no score
SCORE_WEIGHT_CONF = 260.0           # peso da confiança mínima no score
SCORE_DISTANCE_PENALTY = 0.10       # penalidade por distância no score
TOP_FRUITS_FOR_PAIR = 6             # número de frutas consideradas no pairing
SINGLE_SPEED_FAST_THRESHOLD = 80    # velocidade mínima para usar params rápidos
SINGLE_SPEED_ANG_THRESHOLD = 120     # velocidade mínima para usar ângulo perpendicular

# Predição temporal
PREDICT_DURATION_FACTOR = 0.5       # fator de predição em duration * fator

FOCUS_WINDOW_X = 90                 # posição X para refoco da janela
FOCUS_WINDOW_Y = 90                 # posição Y para refoco da janela

# Segurança de bombas
SAFE_BASE_PX = 90                  # distância base segura de bombas
SAFE_PREDICT_BASE_PX = 85         # distância base segura de bombas na predição
SAFE_BOMB_DIAG_FACTOR = 0.6      # fator do raio seguro baseado na diagonal da bomba
SAFE_PREDICT_DIAG_FACTOR = 0.7  # fator do raio seguro na predição baseado na diagonal da bomba


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


def _nearest_match_velocity(cur_pts, prev_pts, dt, *, max_dist=140):
    if dt <= 1e-6 or not prev_pts:
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

    # Shared state
    lock = threading.Lock()
    latest_dets = []
    latest_region = None
    latest_ts = 0.0        # timestamp do FRAME (não do fim da inferência)
    latest_seq = 0
    
    predictor = YoloOnnxPredictor(
        onnx_path=onnx_path,
        imgsz=640,
        conf_thres=MIN_FRUIT_CONF,
        iou_thres=0.45,
        class_agnostic=False,
    )

    # Parâmetros da ação
    single_params_fast = dict(
        length=120,
        down_wait=0.008, # antigo 0.01
        duration=0.04, # antigo 0.05
        steps=2, # antigo 2
    )

    # quando NÃO tem velocidade (1ª aparição), faz um slice mais "generoso"
    single_params_unknown = dict(
        length=150,
        down_wait=0.015, # antigo 0.02
        duration=0.055, # antigo 0.07
        steps=2, # antigo 2
    )

    pair_params = dict(
        down_wait=0.015, # antigo 0.02
        duration=0.055, # antigo 0.07
        steps=3, # antigo 3
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

        recent_boxes = []  # (x1, y1, x2, y2, t)

        def prune_recent_boxes(now):
            recent_boxes[:] = [
                (x1, y1, x2, y2, t)
                for (x1, y1, x2, y2, t) in recent_boxes
                if (now - t) <= RECENT_TTL_S
            ]

        def _recent_center_match(box, ref_box, *, center_thr):
            ax1, ay1, ax2, ay2 = box
            bx1, by1, bx2, by2 = ref_box
            acx = 0.5 * (ax1 + ax2)
            acy = 0.5 * (ay1 + ay2)
            bcx = 0.5 * (bx1 + bx2)
            bcy = 0.5 * (by1 + by2)
            dx = acx - bcx
            dy = acy - bcy
            return (dx * dx + dy * dy) <= (center_thr * center_thr)

        def is_recent_box(x1, y1, x2, y2, now, *, iou_thr, center_thr):
            prune_recent_boxes(now)
            box = (x1, y1, x2, y2)
            for rx1, ry1, rx2, ry2, rt in recent_boxes:
                ref_box = (rx1, ry1, rx2, ry2)
                if _box_iou(box, ref_box) >= iou_thr:
                    return True
                if _recent_center_match(box, ref_box, center_thr=center_thr):
                    return True
            return False

        def add_recent_box(x1, y1, x2, y2, now):
            recent_boxes.append((x1, y1, x2, y2, now))
            prune_recent_boxes(now)

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
            age = now - ts  # ts é do frame
            if age > MAX_DET_AGE_S:
                # det velha -> não age (evita cortar "no passado")
                time.sleep(0.002)
                continue
            
            prune_recent_boxes(now)
            
            # cooldown global
            if (now - last_action_t) < MIN_ACTION_INTERVAL_S:
                time.sleep(0.001)
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
                x1, y1, x2, y2 = _det_xyxy(d)
                boosted_min_area = MIN_FRUIT_AREA
                if is_recent_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    now,
                    iou_thr=RECENT_IOU_THR,
                    center_thr=RECENT_CENTER_PX,
                ):
                    boosted_min_area = int(MIN_FRUIT_AREA * RECENT_AREA_BOOST)
                if conf < MIN_FRUIT_CONF:
                    continue
                if area < boosted_min_area:
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
                dtv = max(1e-6, ts - prev_ts)

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
                    x=FOCUS_WINDOW_X,
                    y=FOCUS_WINDOW_Y,
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
            best_score = -1e18
            top = fruits_scored[:TOP_FRUITS_FOR_PAIR]

            if len(top) >= 2:
                for a in range(len(top)):
                    for b in range(a + 1, len(top)):
                        _, confa, ia, da, cxa, cya, vxa, vya = top[a]
                        _, confb, ib, db, cxb, cyb, vxb, vyb = top[b]

                        dist_ab = math.hypot(cxb - cxa, cyb - cya)
                        if dist_ab < MIN_PAIR_DISTANCE_PX or dist_ab > MAX_PAIR_DISTANCE_PX:
                            continue

                        t_pred = age + pair_params["down_wait"] + pair_params["duration"] * PREDICT_DURATION_FACTOR
                        ax, ay = predict_point(cxa, cya, vxa, vya, t_pred)
                        bx, by = predict_point(cxb, cyb, vxb, vyb, t_pred)

                        if not clamp_inside_window(ax, ay, region.width, region.height, WINDOW_MARGIN_PX):
                            continue
                        if not clamp_inside_window(bx, by, region.width, region.height, WINDOW_MARGIN_PX):
                            continue

                        ax1, ay1, ax2, ay2 = _det_xyxy(da)
                        bx1, by1, bx2, by2 = _det_xyxy(db)
                        if is_recent_box(
                            ax1,
                            ay1,
                            ax2,
                            ay2,
                            now,
                            iou_thr=RECENT_IOU_THR,
                            center_thr=RECENT_CENTER_PX,
                        ):
                            continue
                        if is_recent_box(
                            bx1,
                            by1,
                            bx2,
                            by2,
                            now,
                            iou_thr=RECENT_IOU_THR,
                            center_thr=RECENT_CENTER_PX,
                        ):
                            continue

                        # checa bombas previstas
                        safe = True
                        for k, bd in enumerate(bombs):
                            bcx, bcy = cur_bomb_pts[k]
                            bvx, bvy = bomb_vels[k]
                            px, py = predict_point(bcx, bcy, bvx, bvy, t_pred)
                            bw, bh = _det_wh(bd)
                            safe_r = max(
                                SAFE_PREDICT_BASE_PX,
                                int(SAFE_PREDICT_DIAG_FACTOR * math.hypot(bw, bh)),
                            )
                            if _dist_point_to_segment(px, py, ax, ay, bx, by) <= safe_r:
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
                            best_pair = (ax, ay, bx, by, dyn_overshoot, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)

            try:
                if best_pair is not None:
                    ax, ay, bx, by, dyn_overshoot, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = best_pair
                    if not last_instant_bomb_check(ax, ay, bx, by):
                        continue

                    controller.slice_segment_in_window(
                        ax, ay, bx, by,
                        ScreenOffset(region.left, region.top),
                        window_w=region.width,
                        window_h=region.height,
                        margin=WINDOW_MARGIN_PX,
                        overshoot=dyn_overshoot,
                        **pair_params
                    )

                    last_action_t = time.time()
                    ra1, ra2, ra3, ra4 = _shift_box_to_pred(da, ax, ay)
                    rb1, rb2, rb3, rb4 = _shift_box_to_pred(db, bx, by)
                    add_recent_box(ra1, ra2, ra3, ra4, last_action_t)
                    add_recent_box(rb1, rb2, rb3, rb4, last_action_t)
                    continue

                # -------------------------
                # Fallback: 1 fruta
                # -------------------------
                _, conf0, i0, d0, cx, cy, vx, vy = fruits_scored[0]

                speed = math.hypot(vx, vy)

                # latência esperada até metade do swipe
                # (age já inclui inferência, porque ts é do frame)
                base_params = single_params_fast if speed > SINGLE_SPEED_FAST_THRESHOLD else single_params_unknown
                w0, h0 = _det_wh(d0)
                dyn_overshoot = max(OVERSHOOT_MIN_PX, int(OVERSHOOT_DIAG_FACTOR * math.hypot(w0, h0)))
                t_pred = age + base_params["down_wait"] + base_params["duration"] * PREDICT_DURATION_FACTOR
                px, py = predict_point(cx, cy, vx, vy, t_pred)

                if not clamp_inside_window(px, py, region.width, region.height, WINDOW_MARGIN_PX):
                    continue

                x1, y1, x2, y2 = _det_xyxy(d0)
                if is_recent_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    now,
                    iou_thr=RECENT_IOU_THR,
                    center_thr=RECENT_CENTER_PX,
                ):
                    continue

                # Direção: perpendicular à velocidade quando disponível
                if speed > SINGLE_SPEED_ANG_THRESHOLD:
                    ang = math.degrees(math.atan2(vy, vx)) + 90.0
                else:
                    ang = -45.0

                # Segmento pra recheck de bomba no último instante
                ax = px - INSTANT_BOMB_SEGMENT_OFFSET_PX
                ay = py + INSTANT_BOMB_SEGMENT_OFFSET_PX
                bx = px + INSTANT_BOMB_SEGMENT_OFFSET_PX
                by = py - INSTANT_BOMB_SEGMENT_OFFSET_PX
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
                rx1, ry1, rx2, ry2 = _shift_box_to_pred(d0, px, py)
                add_recent_box(rx1, ry1, rx2, ry2, last_action_t)

            except pyautogui.FailSafeException:
                state.shutdown.set()
                break
            except Exception as e:
                print("Erro no action_worker:", repr(e))
                time.sleep(0.02)

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


def main():
    state = BotState()
    bot_loop(state)


if __name__ == "__main__":
    main()
