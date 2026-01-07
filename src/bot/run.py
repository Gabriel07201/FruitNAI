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
    if ab2 <= 1e-9:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _segment_is_bomb_safe(ax, ay, bx, by, bombs, *, base_safe_px: int = 45) -> bool:
    for b in bombs:
        cx, cy = _det_center(b)
        bw, bh = _det_wh(b)
        safe = max(base_safe_px, int(0.35 * math.hypot(bw, bh)))
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

    predictor = YoloOnnxPredictor(
        onnx_path=onnx_path,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        class_agnostic=False,
    )

    controller = Controller(pause=0.0, fail_safe=True)

    # Shared state
    lock = threading.Lock()
    latest_dets = []
    latest_region = None
    latest_ts = 0.0        # timestamp do FRAME (não do fim da inferência)
    latest_seq = 0

    # -------------------------
    # TUNING PRINCIPAL
    # -------------------------
    max_det_age_s = 0.035         # <- AQUI você deixa o "age max" mais rígido (0.025~0.045)
    min_action_interval_s = 0.085 # intervalo mínimo entre swipes (evita spam)
    focus_every_s = 2.5

    min_fruit_conf = 0.35         # reduz falso positivo que vira "corte a mais"
    min_fruit_area = 800          # ignora dets muito pequenas (metades/efeitos)
    recent_ttl_s = 0.16           # bloqueia repetir corte no mesmo lugar por um curto período
    recent_radius_px = 85

    # Parâmetros da ação
    single_params_fast = dict(
        length=150,
        overshoot=65,
        down_wait=0.04,
        duration=0.09,
        steps=4,
    )

    # quando NÃO tem velocidade (1ª aparição), faz um slice mais "generoso"
    single_params_unknown = dict(
        length=200,
        overshoot=80,
        down_wait=0.04,
        duration=0.10,
        steps=4,
    )

    pair_params = dict(
        overshoot=70,
        down_wait=0.04,
        duration=0.10,
        steps=5,
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
            recent_targets[:] = [(x, y, t) for (x, y, t) in recent_targets if (now - t) <= recent_ttl_s]

        def is_recent(x, y, now):
            prune_recent(now)
            for rx, ry, rt in recent_targets:
                if (x - rx) * (x - rx) + (y - ry) * (y - ry) <= (recent_radius_px * recent_radius_px):
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
            return _segment_is_bomb_safe(ax, ay, bx, by, bombs2, base_safe_px=50)

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
            if age > max_det_age_s:
                # det velha -> não age (evita cortar "no passado")
                time.sleep(0.002)
                continue

            # cooldown global
            if (now - last_action_t) < min_action_interval_s:
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
                if conf < min_fruit_conf:
                    continue
                if area < min_fruit_area:
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
            if (now - last_focus_t) >= focus_every_s:
                controller.focus_window(
                    ScreenOffset(region.left, region.top),
                    window_w=region.width,
                    window_h=region.height,
                    margin=14,
                    x=90,
                    y=90,
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
            top = fruits_scored[:6]

            if len(top) >= 2:
                for a in range(len(top)):
                    for b in range(a + 1, len(top)):
                        _, confa, ia, da, cxa, cya, vxa, vya = top[a]
                        _, confb, ib, db, cxb, cyb, vxb, vyb = top[b]

                        dist_ab = math.hypot(cxb - cxa, cyb - cya)
                        if dist_ab < 55 or dist_ab > 320:
                            continue

                        t_pred = age + pair_params["down_wait"] + pair_params["duration"] * 0.5
                        ax, ay = predict_point(cxa, cya, vxa, vya, t_pred)
                        bx, by = predict_point(cxb, cyb, vxb, vyb, t_pred)

                        if not clamp_inside_window(ax, ay, region.width, region.height, 14):
                            continue
                        if not clamp_inside_window(bx, by, region.width, region.height, 14):
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
                            safe_r = max(50, int(0.38 * math.hypot(bw, bh)))
                            if _dist_point_to_segment(px, py, ax, ay, bx, by) <= safe_r:
                                safe = False
                                break
                        if not safe:
                            continue

                        avg_y = 0.5 * (ay + by)
                        cmin = min(confa, confb)
                        score = (avg_y * 1.0) + (cmin * 260.0) - (dist_ab * 0.10)

                        if score > best_score:
                            best_score = score
                            best_pair = (ax, ay, bx, by, midx, midy)

            try:
                if best_pair is not None:
                    ax, ay, bx, by, midx, midy = best_pair

                    if not last_instant_bomb_check(ax, ay, bx, by):
                        continue

                    controller.slice_segment_in_window(
                        ax, ay, bx, by,
                        ScreenOffset(region.left, region.top),
                        window_w=region.width,
                        window_h=region.height,
                        margin=14,
                        **pair_params
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
                base_params = single_params_fast if speed > 40 else single_params_unknown
                t_pred = age + base_params["down_wait"] + base_params["duration"] * 0.5
                px, py = predict_point(cx, cy, vx, vy, t_pred)

                if not clamp_inside_window(px, py, region.width, region.height, 14):
                    continue

                if is_recent(px, py, now):
                    continue

                # Direção: perpendicular à velocidade quando disponível
                if speed > 60:
                    ang = math.degrees(math.atan2(vy, vx)) + 90.0
                else:
                    ang = -45.0

                # Segmento pra recheck de bomba no último instante
                ax = px - 70
                ay = py + 70
                bx = px + 70
                by = py - 70
                if not last_instant_bomb_check(ax, ay, bx, by):
                    continue

                controller.slice_in_window(
                    px, py,
                    ScreenOffset(region.left, region.top),
                    window_w=region.width,
                    window_h=region.height,
                    margin=14,
                    angle_deg=ang,
                    **base_params
                )

                last_action_t = time.time()
                add_recent(px, py, last_action_t)

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
