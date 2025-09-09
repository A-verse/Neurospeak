"""
ui/display_output.py - NeuroSpeak Display System (~400 lines)
Thread-safe Tkinter UI for phrase output with themes, animations, resizing.
"""

import tkinter as tk
from tkinter import font as tkfont
import threading, queue, time, json, os
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Optional
from pathlib import Path
import logging

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DisplayOutput")

# =========================
# Config
# =========================
CONFIG_FILE = "config/display_settings.json"
DEFAULT_DURATION = 5
MIN_WIDTH, MIN_HEIGHT = 300, 80
MAX_WIDTH, MAX_HEIGHT = 800, 200

# =========================
# Theme Enum
# =========================
class Theme(Enum):
    DARK = auto()
    LIGHT = auto()
    HIGH_CONTRAST = auto()

# =========================
# Display Settings
# =========================
@dataclass
class DisplaySettings:
    duration: int = DEFAULT_DURATION
    theme: Theme = Theme.DARK
    font_size: int = 18
    font_family: str = "Arial"
    bold_text: bool = True
    opacity: float = 1.0
    always_on_top: bool = True
    window_width: int = 500
    window_height: int = 100
    show_speaker: bool = True
    animation_speed: int = 2  # 0-5 scale

# =========================
# Utility: Theme Colors
# =========================
def get_colors(theme: Theme):
    if theme == Theme.DARK:
        return {"bg": "#121212", "text": "#FFFFFF", "speaker": "#BBBBBB", "status_bg": "#222222", "status_text": "#AAAAAA"}
    if theme == Theme.LIGHT:
        return {"bg": "#F5F5F5", "text": "#000000", "speaker": "#555555", "status_bg": "#E0E0E0", "status_text": "#666666"}
    if theme == Theme.HIGH_CONTRAST:
        return {"bg": "#000000", "text": "#00FF00", "speaker": "#FFFF00", "status_bg": "#000000", "status_text": "#FFFFFF"}
    return {"bg": "#121212", "text": "#FFFFFF", "speaker": "#BBBBBB", "status_bg": "#222222", "status_text": "#AAAAAA"}

# =========================
# Display Window Class
# =========================
class DisplayWindow:
    def __init__(self, q: queue.Queue):
        self.queue = q
        self.settings = self._load_settings()
        self.colors = get_colors(self.settings.theme)

        self.window = tk.Tk()
        self.window.title("NeuroSpeak Output")
        self.window.geometry(f"{self.settings.window_width}x{self.settings.window_height}")
        self.window.configure(bg=self.colors["bg"])
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.withdraw()

        self.window.attributes("-alpha", self.settings.opacity)
        self.window.attributes("-topmost", self.settings.always_on_top)

        self._setup_ui()
        self._setup_bindings()
        self._check_queue()

        self.current_animation = None
        self._positioned = False

    # ---------- Settings ----------
    def _load_settings(self) -> DisplaySettings:
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    return DisplaySettings(**data)
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
        return DisplaySettings()

    def _save_settings(self):
        try:
            Path("config").mkdir(exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(asdict(self.settings), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    # ---------- UI Setup ----------
    def _setup_ui(self):
        weight = "bold" if self.settings.bold_text else "normal"
        self.font = tkfont.Font(family=self.settings.font_family, size=self.settings.font_size, weight=weight)

        # Main frame
        self.text_frame = tk.Frame(self.window, bg=self.colors["bg"], padx=10, pady=10)
        self.text_frame.pack(expand=True, fill=tk.BOTH)

        # Speaker label
        self.speaker_label = tk.Label(self.text_frame, text="", font=(self.settings.font_family, self.settings.font_size-4),
                                      bg=self.colors["bg"], fg=self.colors["speaker"], anchor="w")
        self.speaker_label.pack(fill=tk.X)

        # Phrase label
        self.content_label = tk.Label(self.text_frame, text="", font=self.font, bg=self.colors["bg"],
                                      fg=self.colors["text"], wraplength=self.settings.window_width-20,
                                      justify=tk.LEFT, anchor="w")
        self.content_label.pack(expand=True, fill=tk.BOTH)

        # Status bar
        self.status_bar = tk.Label(self.window, text="", bg=self.colors["status_bg"], fg=self.colors["status_text"], anchor="e", padx=5)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_bindings(self):
        self.window.bind("<Configure>", self._on_resize)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Event Handlers ----------
    def _on_resize(self, event):
        if event.widget == self.window:
            width = max(MIN_WIDTH, min(event.width, MAX_WIDTH))
            height = max(MIN_HEIGHT, min(event.height, MAX_HEIGHT))
            self.settings.window_width = width
            self.settings.window_height = height
            self.content_label.config(wraplength=width-20)
            self._save_settings()

    def _on_close(self):
        self.window.withdraw()

    # ---------- Queue Processing ----------
    def _check_queue(self):
        try:
            while not self.queue.empty():
                phrase, speaker = self.queue.get_nowait()
                self._show_text(phrase, speaker)
        except queue.Empty:
            pass
        self.window.after(100, self._check_queue)

    # ---------- Animations ----------
    def _animate_show(self):
        if self.current_animation:
            self.window.after_cancel(self.current_animation)
        opacity = self.window.attributes("-alpha")
        if opacity < self.settings.opacity:
            new_opacity = min(self.settings.opacity, opacity + 0.1)
            self.window.attributes("-alpha", new_opacity)
            self.current_animation = self.window.after(30, self._animate_show)

    def _animate_hide(self):
        if self.current_animation:
            self.window.after_cancel(self.current_animation)
        opacity = self.window.attributes("-alpha")
        if opacity > 0.1:
            new_opacity = max(0.1, opacity - 0.1)
            self.window.attributes("-alpha", new_opacity)
            self.current_animation = self.window.after(30, self._animate_hide)
        else:
            self.window.withdraw()
            self.current_animation = None

    # ---------- Display ----------
    def _show_text(self, text: str, speaker: Optional[str]):
        self.content_label.config(text=text)
        self.speaker_label.config(text=f"Speaker: {speaker}" if speaker and self.settings.show_speaker else "")
        self.status_bar.config(text=time.strftime("%H:%M:%S"))

        self.window.deiconify()
        self.window.lift()
        self._animate_show()

        if not self._positioned:
            self._center()
            self._positioned = True

        self.window.after(self.settings.duration * 1000, self._animate_hide)

    def _center(self):
        self.window.update_idletasks()
        w, h = self.window.winfo_width(), self.window.winfo_height()
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        x, y = (sw//2 - w//2), (sh//3 - h//2)
        self.window.geometry(f"+{x}+{y}")

    def run(self):
        self.window.mainloop()

# =========================
# Singleton & Public API
# =========================
_display_instance = None
_display_thread = None
_queue = queue.Queue()

def init_display():
    global _display_instance, _display_thread
    if not _display_instance:
        _display_instance = DisplayWindow(_queue)
        _display_thread = threading.Thread(target=_display_instance.run, daemon=True)
        _display_thread.start()
        logger.info("✅ Display initialized")

def show_phrase(phrase: str, speaker: Optional[str] = None):
    _queue.put((phrase, speaker))
