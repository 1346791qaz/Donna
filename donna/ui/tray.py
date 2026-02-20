"""
ui/tray.py — Windows system tray icon and menu for Donna.

Uses pystray with a Pillow-generated icon.
The tray icon provides:
  - Show / Hide the floating window
  - Mute / Unmute microphone
  - Exit
"""

import logging
import threading
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw
import pystray

logger = logging.getLogger(__name__)


def _create_default_icon(size: int = 64) -> Image.Image:
    """Draw a simple 'D' on a dark purple circle as a fallback icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Circle background
    draw.ellipse([0, 0, size - 1, size - 1], fill=(75, 0, 130, 255))
    # Letter 'D'
    draw.text((size // 4, size // 8), "D", fill=(255, 255, 255, 255))
    return img


def _load_icon_image() -> Image.Image:
    icon_path = Path(__file__).parent.parent.parent / "assets" / "donna_icon.png"
    if icon_path.exists():
        return Image.open(str(icon_path))
    return _create_default_icon()


class SystemTray:
    """Manages the Windows system tray icon and context menu."""

    def __init__(
        self,
        on_show_window: Callable[[], None],
        on_hide_window: Callable[[], None],
        on_exit: Callable[[], None],
        on_toggle_mute: Callable[[], None],
    ):
        self._on_show = on_show_window
        self._on_hide = on_hide_window
        self._on_exit = on_exit
        self._on_toggle_mute = on_toggle_mute
        self._muted = False
        self._icon: pystray.Icon | None = None

    def _make_menu(self) -> pystray.Menu:
        mute_label = "Unmute Microphone" if self._muted else "Mute Microphone"
        return pystray.Menu(
            pystray.MenuItem("Donna", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Window", self._handle_show),
            pystray.MenuItem("Hide Window", self._handle_hide),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(mute_label, self._handle_toggle_mute),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._handle_exit),
        )

    def _refresh_menu(self) -> None:
        if self._icon:
            self._icon.menu = self._make_menu()
            self._icon.update_menu()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_show(self, icon, item) -> None:
        self._on_show()

    def _handle_hide(self, icon, item) -> None:
        self._on_hide()

    def _handle_toggle_mute(self, icon, item) -> None:
        self._muted = not self._muted
        self._on_toggle_mute()
        self._refresh_menu()

    def _handle_exit(self, icon, item) -> None:
        icon.stop()
        self._on_exit()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the tray icon.  pystray.Icon.run() blocks, so this must be
        called from a dedicated thread.
        """
        image = _load_icon_image()
        self._icon = pystray.Icon(
            name="Donna",
            icon=image,
            title="Donna — Personal Assistant",
            menu=self._make_menu(),
        )
        logger.info("System tray icon starting.")
        self._icon.run()

    def start_threaded(self) -> threading.Thread:
        """Launch the tray in a background daemon thread."""
        t = threading.Thread(target=self.start, daemon=True, name="SystemTray")
        t.start()
        return t

    def update_title(self, title: str) -> None:
        if self._icon:
            self._icon.title = title

    def notify(self, title: str, message: str) -> None:
        """Show a Windows toast notification from the tray icon."""
        if self._icon:
            try:
                self._icon.notify(message, title)
            except Exception:
                logger.debug("Tray notification not supported on this platform.")

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
