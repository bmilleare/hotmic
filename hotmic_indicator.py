#!/usr/bin/env python3
"""Pulsing recording indicator overlay for dictation mode."""

import math
import signal

import gi

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, GLib, Gtk

import cairo

BADGE_W = 200
BADGE_H = 54
CORNER_R = 14
DOT_BASE_R = 9
DOT_PULSE_R = 3
FPS = 30


class RecordingIndicator(Gtk.Window):
    def __init__(self):
        super().__init__()
        self.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_accept_focus(False)
        self.set_can_focus(False)
        self.set_app_paintable(True)
        self.set_resizable(False)
        self.stick()
        self.set_size_request(BADGE_W, BADGE_H)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)

        # Position: top-center, just below any panel
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0)
        work = monitor.get_workarea()
        self.move(work.x + work.width // 2 - BADGE_W // 2, work.y + 8)

        self.connect("realize", self._make_click_through)

        self.phase = 0.0
        self.start_us = GLib.get_monotonic_time()
        self.connect("draw", self.on_draw)
        GLib.timeout_add(1000 // FPS, self.on_tick)
        self.show_all()

    def _make_click_through(self, _widget):
        region = cairo.Region(cairo.RectangleInt(0, 0, 0, 0))
        self.get_window().input_shape_combine_region(region, 0, 0)

    def on_tick(self):
        self.phase += 0.07
        self.queue_draw()
        return True

    def on_draw(self, _widget, cr):
        # Clear to transparent
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        p = (math.sin(self.phase) + 1) / 2  # 0..1

        # Dark rounded-rect background
        _rounded_rect(cr, 0, 0, BADGE_W, BADGE_H, CORNER_R)
        cr.set_source_rgba(0.12, 0.12, 0.12, 0.88)
        cr.fill()

        # Pulsing red border
        _rounded_rect(cr, 0.5, 0.5, BADGE_W - 1, BADGE_H - 1, CORNER_R)
        cr.set_source_rgba(0.8, 0.1, 0.1, 0.35 + p * 0.35)
        cr.set_line_width(1.5)
        cr.stroke()

        # --- Red dot with glow ---
        dot_x, dot_y = 30, BADGE_H / 2
        r = DOT_BASE_R + p * DOT_PULSE_R
        a = 0.5 + p * 0.5

        # Outer glow
        cr.set_source_rgba(1, 0, 0, a * 0.18)
        cr.arc(dot_x, dot_y, r + 9, 0, 2 * math.pi)
        cr.fill()

        # Main dot
        cr.set_source_rgba(1, 0.08, 0.08, a)
        cr.arc(dot_x, dot_y, r, 0, 2 * math.pi)
        cr.fill()

        # Specular highlight
        cr.set_source_rgba(1, 0.5, 0.5, a * 0.4)
        cr.arc(dot_x - r * 0.22, dot_y - r * 0.22, r * 0.35, 0, 2 * math.pi)
        cr.fill()

        # --- "REC" label ---
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(19)
        cr.set_source_rgba(1, 1, 1, 0.92)
        rec_ext = cr.text_extents("REC")
        baseline_y = BADGE_H / 2 + rec_ext.height / 2
        cr.move_to(50, baseline_y)
        cr.show_text("REC")

        # --- Elapsed timer ---
        elapsed_s = int((GLib.get_monotonic_time() - self.start_us) / 1_000_000)
        mm, ss = divmod(elapsed_s, 60)
        timer = f"{mm}:{ss:02d}"
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)
        cr.set_source_rgba(1, 1, 1, 0.50)
        cr.move_to(50 + rec_ext.x_advance + 12, baseline_y)
        cr.show_text(timer)

        return True


def _rounded_rect(cr, x, y, w, h, r):
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def _quit(*_args):
    Gtk.main_quit()
    return GLib.SOURCE_REMOVE


def main():
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _quit)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _quit)
    RecordingIndicator()
    Gtk.main()


if __name__ == "__main__":
    main()
