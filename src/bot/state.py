from dataclasses import dataclass, field
import threading

@dataclass
class BotState:
    running: threading.Event = field(default_factory=threading.Event)  # start/stop
    shutdown: threading.Event = field(default_factory=threading.Event) # sair de vez
