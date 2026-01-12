from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

import pyautogui


@dataclass(frozen=True)
class ScreenOffset:
    left: int
    top: int


class Controller:
    """
    Controle simples por mouse/teclado.
    Para alguns jogos (principalmente DirectX), pode ser necessário trocar pyautogui por pydirectinput.
    """

    def __init__(self, pause: float = 0.0, fail_safe: bool = True):
        pyautogui.PAUSE = pause
        pyautogui.FAILSAFE = fail_safe

    # Helpers de clamp e coords
    def _clamp_rel_xy(
        self,
        x: int,
        y: int,
        window_w: Optional[int],
        window_h: Optional[int],
        margin: int,
    ) -> tuple[int, int]:
        if window_w is None or window_h is None:
            return x, y
        x = max(margin, min(window_w - 1 - margin, x))
        y = max(margin, min(window_h - 1 - margin, y))
        return x, y

    def _abs_xy(
        self,
        x: int,
        y: int,
        offset: ScreenOffset,
        window_w: Optional[int] = None,
        window_h: Optional[int] = None,
        margin: int = 12,
    ) -> tuple[int, int]:
        rx, ry = self._clamp_rel_xy(x, y, window_w, window_h, margin)
        return offset.left + rx, offset.top + ry

    # Ações básicas
    def click_in_window(self, x: int, y: int, offset: ScreenOffset, button: str = "left"):
        pyautogui.click(x=offset.left + x, y=offset.top + y, button=button)

    def focus_window(
        self,
        offset: ScreenOffset,
        *,
        window_w: Optional[int] = None,
        window_h: Optional[int] = None,
        margin: int = 12,
        x: int = 80,
        y: int = 80,
    ):
        """
        Dá foco no jogo com um clique curto num ponto "seguro" (canto).
        Faz clamp para não clicar fora da janela.
        """
        ax, ay = self._abs_xy(x, y, offset, window_w, window_h, margin)
        pyautogui.click(ax, ay, button="left")

    def drag_in_window(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        offset: ScreenOffset,
        duration: float = 0.10,
        button: str = "left",
    ):
        pyautogui.moveTo(offset.left + x1, offset.top + y1)
        pyautogui.dragTo(offset.left + x2, offset.top + y2, duration=duration, button=button)

    def press(self, key: str):
        pyautogui.press(key)

    # Slice (linha / segmento)
    def slice_line_in_window(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        offset: ScreenOffset,
        *,
        window_w: Optional[int] = None,
        window_h: Optional[int] = None,
        margin: int = 12,
        down_wait: float = 0.06,
        duration: float = 0.18,
        settle_wait: float = 0.01,
        steps: int = 0,
    ):
        """
        Faz um "slice" segurando o botão do mouse por um tempo e movendo até o destino.
        Com clamp para não sair da janela do jogo.
        """
        ax1, ay1 = self._abs_xy(x1, y1, offset, window_w, window_h, margin)
        ax2, ay2 = self._abs_xy(x2, y2, offset, window_w, window_h, margin)

        pyautogui.moveTo(ax1, ay1)
        pyautogui.mouseDown(button="left")
        if down_wait > 0:
            time.sleep(down_wait)

        if steps and steps > 0:
            for i in range(1, steps + 1):
                t = i / steps
                mx = int(ax1 + (ax2 - ax1) * t)
                my = int(ay1 + (ay2 - ay1) * t)

                # clamp também no caminho (importante quando o overshoot encosta em borda)
                if window_w is not None and window_h is not None:
                    rx = mx - offset.left
                    ry = my - offset.top
                    rx, ry = self._clamp_rel_xy(rx, ry, window_w, window_h, margin)
                    mx, my = offset.left + rx, offset.top + ry

                pyautogui.moveTo(mx, my)
                if duration > 0:
                    time.sleep(duration / steps)
        else:
            pyautogui.moveTo(ax2, ay2, duration=max(0.0, duration))

        if settle_wait > 0:
            time.sleep(settle_wait)

        pyautogui.mouseUp(button="left")

    def slice_in_window(
        self,
        x: int,
        y: int,
        offset: ScreenOffset,
        *,
        window_w: Optional[int] = None,
        window_h: Optional[int] = None,
        margin: int = 12,
        dx: int | None = None,
        dy: int | None = None,
        length: int = 180,
        angle_deg: float = -45.0,
        overshoot: int = 40,
        down_wait: float = 0.08,
        duration: float = 0.22,
        steps: int = 0,
    ):
        """
        Faz um slice que atravessa o ponto (x,y).
        Clamp garante que não saia da janela.
        """
        if dx is not None or dy is not None:
            ddx = int(dx or 0)
            ddy = int(dy or 0)
            if ddx == 0 and ddy == 0:
                ddx, ddy = length, 0
        else:
            rad = math.radians(angle_deg)
            ddx = int(math.cos(rad) * length)
            ddy = int(math.sin(rad) * length)

        mag = math.hypot(ddx, ddy)
        if mag < 1e-6:
            ddx, ddy = length, 0
            mag = float(length)

        ux = ddx / mag
        uy = ddy / mag

        half = 0.5 * mag
        extra = float(overshoot)

        x1 = int(x - ux * (half + extra))
        y1 = int(y - uy * (half + extra))
        x2 = int(x + ux * (half + extra))
        y2 = int(y + uy * (half + extra))

        self.slice_line_in_window(
            x1, y1, x2, y2, offset,
            window_w=window_w,
            window_h=window_h,
            margin=margin,
            down_wait=down_wait,
            duration=duration,
            steps=steps,
        )
        
    def slice_short_from_center_in_window(
        self,
        x: int,
        y: int,
        offset: ScreenOffset,
        *,
        window_w: Optional[int] = None,
        window_h: Optional[int] = None,
        margin: int = 12,
        dx: int | None = None,
        dy: int | None = None,
        length: int = 80,
        angle_deg: float = -45.0,
        down_wait: float = 0.04,
        duration: float = 0.08,
        steps: int = 1,
    ):
        """
        Slice curto a partir do centro (x, y) até um ponto na direção desejada.
        """
        if dx is not None or dy is not None:
            ddx = float(dx or 0)
            ddy = float(dy or 0)
            if ddx == 0 and ddy == 0:
                ddx, ddy = float(length), 0.0
        else:
            rad = math.radians(angle_deg)
            ddx = math.cos(rad)
            ddy = math.sin(rad)

        mag = math.hypot(ddx, ddy)
        if mag < 1e-6:
            ddx, ddy = 1.0, 0.0
            mag = 1.0

        ux = ddx / mag
        uy = ddy / mag
        x2 = int(x + ux * length)
        y2 = int(y + uy * length)

        self.slice_line_in_window(
            x, y, x2, y2, offset,
            window_w=window_w,
            window_h=window_h,
            margin=margin,
            down_wait=down_wait,
            duration=duration,
            steps=steps,
        )

    def slice_segment_in_window(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        offset: ScreenOffset,
        *,
        window_w: Optional[int] = None,
        window_h: Optional[int] = None,
        margin: int = 12,
        overshoot: int = 60,
        max_overshoot_frac: float | None = 0.45,
        down_wait: float = 0.07,
        duration: float = 0.18,
        steps: int = 6,
    ):
        """
        Slice de um ponto a outro (ex.: centro de fruta A -> centro de fruta B),
        com overshoot para atravessar bem as frutas.
        """
        vx = float(x2 - x1)
        vy = float(y2 - y1)
        mag = math.hypot(vx, vy)
        if mag < 1e-6:
            # cai num slice curto se os pontos forem iguais
            self.slice_in_window(
                x1, y1, offset,
                window_w=window_w,
                window_h=window_h,
                margin=margin,
                overshoot=overshoot,
                down_wait=down_wait,
                duration=duration,
                steps=steps,
            )
            return

        ux = vx / mag
        uy = vy / mag
        
        if max_overshoot_frac is not None and max_overshoot_frac > 0:
            overshoot = min(overshoot, int(mag * max_overshoot_frac))

        sx = int(x1 - ux * overshoot)
        sy = int(y1 - uy * overshoot)
        ex = int(x2 + ux * overshoot)
        ey = int(y2 + uy * overshoot)

        self.slice_line_in_window(
            sx, sy, ex, ey, offset,
            window_w=window_w,
            window_h=window_h,
            margin=margin,
            down_wait=down_wait,
            duration=duration,
            steps=steps,
        )
