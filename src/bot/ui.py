import threading
import tkinter as tk
from tkinter import ttk

from .state import BotState
from .run import bot_loop


def main():
    state = BotState()

    # Thread do bot
    t = threading.Thread(target=bot_loop, args=(state,), daemon=True)
    t.start()

    # Janela
    root = tk.Tk()
    root.title("FruitNAI")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    status_var = tk.StringVar(value="Parado")

    def update_status():
        status_var.set("Rodando" if state.running.is_set() else "Parado")

    def start():
        state.running.set()
        update_status()

    def stop():
        state.running.clear()
        update_status()

    def toggle_running(event=None):
        if state.running.is_set():
            stop()
        else:
            start()

    def quit_app():
        state.running.clear()
        state.shutdown.set()
        root.destroy()

    frm = ttk.Frame(root, padding=12)
    frm.grid()

    ttk.Label(frm, text="Controle do Bot").grid(row=0, column=0, columnspan=3, pady=(0, 8))

    root.bind("<F2>", toggle_running)
    ttk.Button(frm, text="Começar/Pausar (F2)", command=stop).grid(row=2, column=1, padx=5)
    ttk.Button(frm, text="Encerrar (F3)", command=quit_app).grid(row=2, column=2, padx=5)

    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
