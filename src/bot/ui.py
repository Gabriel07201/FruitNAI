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

    def start():
        state.running.set()
        status_var.set("Rodando")

    def stop():
        state.running.clear()
        status_var.set("Parado")

    def quit_app():
        state.running.clear()
        state.shutdown.set()
        root.destroy()

    frm = ttk.Frame(root, padding=12)
    frm.grid()

    ttk.Label(frm, text="Controle do Bot").grid(row=0, column=0, columnspan=3, pady=(0, 8))
    ttk.Label(frm, textvariable=status_var).grid(row=1, column=0, columnspan=3, pady=(0, 10))

    ttk.Button(frm, text="Iniciar", command=start).grid(row=2, column=0, padx=5)
    ttk.Button(frm, text="Parar", command=stop).grid(row=2, column=1, padx=5)
    ttk.Button(frm, text="Sair", command=quit_app).grid(row=2, column=2, padx=5)

    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
