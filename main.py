"""
main.py
───────
Entry‑point for the Carrot Defect Detection desktop application.

Launches a warm, light‑themed CustomTkinter GUI with:
  • Card‑based split dashboard (preview left, controls right)
  • Animated carrot watermark background
  • Image upload, preview toggle, stats, severity grading
  • Download Quality Report (.txt export)
  • Export annotated‑image button
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, Canvas
from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from detection_pipeline import run_pipeline, PipelineResult
from severity_index import compute_severity, SeverityReport

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Appearance ───────────────────────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# ── Paths ────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _APP_DIR / "model" / "carrot_defect_mobilenetv2.tflite"
_SUPPORTED_EXTS = (
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
    ("All files", "*.*"),
)

# ── Colour palette ───────────────────────────────────────────────────
_BG_PEACH     = "#FFF0E0"
_BG_WARM      = "#FDE8D0"
_CARD_WHITE   = "#FFFFFF"
_CARD_SHADOW  = "#F0E0D0"

_ORANGE       = "#FF8C42"
_ORANGE_DARK  = "#E07030"
_ORANGE_LIGHT = "#FFA564"
_YELLOW       = "#FFD166"
_YELLOW_DARK  = "#F0C050"
_GREEN_OK     = "#4CAF50"
_RED_ALERT    = "#EF4444"

_TEXT_DARK    = "#3D2B1F"
_TEXT_MED     = "#7A6050"
_TEXT_LIGHT   = "#A08878"
_TEXT_WHITE   = "#FFFFFF"

_WATERMARK_COLOR = "#F5D8B8"


# =====================================================================
#  Watermark Canvas  – faint carrot with breathing animation
# =====================================================================

class WatermarkCanvas(Canvas):
    """Draws a large, faint carrot silhouette that gently 'breathes'."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            highlightthickness=0,
            bg=_BG_PEACH,
            **kwargs,
        )
        self._phase = 0.0
        self._items: list[int] = []
        self.bind("<Configure>", self._on_resize)
        self._animate()

    # -- draw the carrot silhouette ------------------------------------

    def _draw(self):
        self.delete("wm")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 50 or h < 50:
            return

        # Breathing scale factor (subtle: 0.97 → 1.03)
        scale = 1.0 + 0.03 * math.sin(self._phase)
        cx, cy = w * 0.50, h * 0.52

        # -- Carrot body (elongated oval) --
        bw = 70 * scale
        bh = 180 * scale
        self.create_oval(
            cx - bw, cy - bh, cx + bw, cy + bh,
            fill=_WATERMARK_COLOR, outline="", tags="wm",
        )
        # Taper / root tip (triangle)
        self.create_polygon(
            cx - bw * 0.55, cy + bh * 0.75,
            cx + bw * 0.55, cy + bh * 0.75,
            cx, cy + bh * 1.45,
            fill=_WATERMARK_COLOR, outline="", tags="wm", smooth=True,
        )

        # -- Leaf stalks --
        for angle_offset, length_f in [(-18, 1.0), (0, 1.18), (16, 0.95)]:
            rad = math.radians(-90 + angle_offset)
            x1 = cx
            y1 = cy - bh * 0.85
            x2 = cx + math.cos(rad) * bh * length_f * 0.6
            y2 = cy - bh * 0.85 + math.sin(rad) * bh * length_f * 0.6
            self.create_line(
                x1, y1, x2, y2,
                fill=_WATERMARK_COLOR, width=8 * scale,
                tags="wm", smooth=True,
            )
            # Leaf blob at tip
            lr = 18 * scale
            self.create_oval(
                x2 - lr, y2 - lr * 1.5, x2 + lr, y2 + lr * 0.5,
                fill=_WATERMARK_COLOR, outline="", tags="wm",
            )

        # -- Horizontal stripes on the body (carrot texture) --
        for frac in [0.15, 0.35, 0.55, 0.72]:
            yy = cy - bh + 2 * bh * frac
            hw = bw * (1.0 - 0.4 * abs(frac - 0.45))
            self.create_line(
                cx - hw * 0.7, yy, cx + hw * 0.7, yy,
                fill="#F0D0A8", width=2, tags="wm",
            )

    # -- animation loop ------------------------------------------------

    def _animate(self):
        self._phase += 0.04
        self._draw()
        self.after(50, self._animate)

    def _on_resize(self, _event=None):
        self._draw()


# =====================================================================
#  Main Application Window
# =====================================================================

class CarrotDetectorApp(ctk.CTk):
    """Top‑level application window – warm light‑themed dashboard."""

    def __init__(self) -> None:
        super().__init__()

        self.title("🥕  Carrot Defect Detector")
        self.geometry("1200x820")
        self.minsize(1020, 680)
        self.configure(fg_color=_BG_PEACH)

        self._source_bgr: Optional[np.ndarray] = None
        self._result: Optional[PipelineResult] = None
        self._severity: Optional[SeverityReport] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._image_path: Optional[str] = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:

        # -- Watermark background layer --------------------------------
        self._watermark = WatermarkCanvas(self)
        self._watermark.place(relx=0, rely=0, relwidth=1, relheight=1)

        # -- Container for all foreground widgets ----------------------
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ── Top bar ──────────────────────────────────────────────────
        top = ctk.CTkFrame(container, fg_color="transparent", height=60)
        top.pack(fill="x", padx=28, pady=(18, 0))

        ctk.CTkLabel(
            top,
            text="🥕",
            font=ctk.CTkFont(size=32),
        ).pack(side="left")

        title_frame = ctk.CTkFrame(top, fg_color="transparent")
        title_frame.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(
            title_frame,
            text="Carrot Defect Detector",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color=_TEXT_DARK,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_frame,
            text="AI‑powered quality inspection & grading",
            font=ctk.CTkFont(size=12),
            text_color=_TEXT_LIGHT,
        ).pack(anchor="w")

        # Version pill
        ctk.CTkLabel(
            top,
            text=" v1.0 ",
            font=ctk.CTkFont(size=11),
            text_color=_ORANGE,
            fg_color=_BG_WARM,
            corner_radius=8,
        ).pack(side="right", padx=4)

        # ── Main body (two‑column split) ─────────────────────────────
        body = ctk.CTkFrame(container, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=28, pady=16)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_preview_card(body)
        self._build_right_panel(body)

    # ── LEFT: Preview card ───────────────────────────────────────────

    def _build_preview_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent, corner_radius=20, fg_color=_CARD_WHITE,
            border_width=1, border_color=_CARD_SHADOW,
        )
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # Header row
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=22, pady=(18, 6))

        ctk.CTkLabel(
            header, text="📷  Image Preview",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=_TEXT_DARK,
        ).pack(side="left")

        self._view_toggle = ctk.CTkSegmentedButton(
            header,
            values=["Original", "Annotated"],
            command=self._on_view_toggle,
            font=ctk.CTkFont(size=12),
            selected_color=_ORANGE,
            selected_hover_color=_ORANGE_DARK,
            unselected_color=_BG_WARM,
            unselected_hover_color=_BG_PEACH,
            text_color=_TEXT_WHITE,
            text_color_disabled=_TEXT_MED,
        )
        self._view_toggle.set("Original")
        self._view_toggle.pack(side="right")

        # Canvas area
        self._canvas_frame = ctk.CTkFrame(
            card, fg_color=_BG_PEACH, corner_radius=16,
        )
        self._canvas_frame.pack(fill="both", expand=True, padx=18, pady=(6, 18))

        self._canvas_label = ctk.CTkLabel(
            self._canvas_frame,
            text="🥕\n\nUpload an image to start inspection",
            font=ctk.CTkFont(size=15),
            text_color=_TEXT_LIGHT,
        )
        self._canvas_label.pack(expand=True)

    # ── RIGHT: Stacked cards ─────────────────────────────────────────

    def _build_right_panel(self, parent) -> None:
        right = ctk.CTkScrollableFrame(
            parent,
            fg_color="transparent",
            scrollbar_button_color=_ORANGE,
            scrollbar_button_hover_color=_ORANGE_DARK,
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        # Card 1 – Upload & Export
        self._build_upload_card(right)

        # Card 2 – Statistics & Grading
        self._build_stats_card(right)

        # Card 3 – Treatment Guidance (High-contrast, dedicated card)
        self._build_treatment_card(right)

    # -- Upload card ---------------------------------------------------

    def _build_upload_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent, corner_radius=20, fg_color=_CARD_WHITE,
            border_width=1.5, border_color=_CARD_SHADOW,
        )
        card.pack(fill="x", pady=(0, 12))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=16)

        self._upload_btn = ctk.CTkButton(
            inner,
            text="📂  Upload Carrot Image",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=_TEXT_WHITE,
            fg_color=_ORANGE,
            hover_color=_ORANGE_DARK,
            height=46,
            corner_radius=14,
            command=self._on_upload,
        )
        self._upload_btn.pack(fill="x")

        # Export row
        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        btn_row.columnconfigure((0, 1), weight=1)

        self._export_btn = ctk.CTkButton(
            btn_row,
            text="💾 Export Image",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=_TEXT_DARK,
            fg_color=_YELLOW,
            hover_color=_YELLOW_DARK,
            height=36,
            corner_radius=12,
            state="disabled",
            command=self._on_export,
        )
        self._export_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self._report_btn = ctk.CTkButton(
            btn_row,
            text="📄 Download Report",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=_TEXT_DARK,
            fg_color=_YELLOW,
            hover_color=_YELLOW_DARK,
            height=36,
            corner_radius=12,
            state="disabled",
            command=self._on_download_report,
        )
        self._report_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    # -- Stats card ----------------------------------------------------

    def _build_stats_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent, corner_radius=20, fg_color=_CARD_WHITE,
            border_width=1.5, border_color=_CARD_SHADOW,
        )
        card.pack(fill="x", pady=(0, 12))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=16)

        ctk.CTkLabel(
            inner, text="📊  Statistics & Grading",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=_TEXT_DARK, anchor="w",
        ).pack(fill="x", pady=(0, 10))

        # Disease / defect name label
        self._disease_label = ctk.CTkLabel(
            inner, text="Awaiting Image Inspection",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=_TEXT_MED,
            corner_radius=12,
            fg_color=_BG_WARM,
            height=38,
        )
        self._disease_label.pack(fill="x", pady=(0, 8))

        # Grade badge
        self._grade_label = ctk.CTkLabel(
            inner, text="Awaiting image…",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=_TEXT_LIGHT,
            corner_radius=12,
            fg_color=_BG_PEACH,
            height=44,
        )
        self._grade_label.pack(fill="x", pady=(0, 12))

        # Metrics grid (2×2)
        grid = ctk.CTkFrame(inner, fg_color="transparent")
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)

        self._stat_defect = self._metric_tile(grid, "Defect Area", "— %", 0, 0)
        self._stat_conf   = self._metric_tile(grid, "Confidence",  "— %", 0, 1)
        self._stat_comp   = self._metric_tile(grid, "Composite",   "—",   1, 0)
        self._stat_model  = self._metric_tile(grid, "Pipeline",    "—",   1, 1)

    def _metric_tile(self, parent, title, value, row, col) -> ctk.CTkLabel:
        tile = ctk.CTkFrame(
            parent, fg_color=_BG_PEACH, corner_radius=14,
        )
        tile.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)

        ctk.CTkLabel(
            tile, text=title,
            font=ctk.CTkFont(size=11), text_color=_TEXT_LIGHT,
        ).pack(padx=10, pady=(8, 0))

        val = ctk.CTkLabel(
            tile, text=value,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=_TEXT_DARK,
        )
        val.pack(padx=10, pady=(2, 8))
        return val

    # -- Treatment card (Restored high-contrast white card) -------------

    def _build_treatment_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent, corner_radius=20, fg_color=_CARD_WHITE,
            border_width=1.5, border_color=_CARD_SHADOW,
        )
        card.pack(fill="x", pady=(0, 8))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=16)

        header_row = ctk.CTkFrame(inner, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            header_row, text="💊  Treatment Guidance",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=_TEXT_DARK, anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            header_row, text="Actionable Precautions",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=_ORANGE_DARK,
            fg_color=_BG_WARM,
            corner_radius=6,
            padx=8, pady=3,
        ).pack(side="right")

        self._treatment_box = ctk.CTkTextbox(
            inner,
            height=280,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#FFFDFB",
            text_color=_TEXT_DARK,
            border_width=1,
            border_color="#F0D9C8",
            corner_radius=12,
            wrap="word",
            activate_scrollbars=True,
        )
        self._treatment_box.pack(fill="both", expand=True)
        self._treatment_box.insert(
            "1.0",
            "📌 Awaiting Image Inspection\n\n"
            "Upload a carrot image using the button above to begin analysis.\n\n"
            "Step-by-step treatment guidance, sanitization wash protocols, batch quarantine, "
            "and cold-storage specifications will be displayed here.",
        )
        self._treatment_box.configure(state="disabled")

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_upload(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a carrot image",
            filetypes=_SUPPORTED_EXTS,
        )
        if not path:
            return

        self._image_path = path
        self._upload_btn.configure(state="disabled", text="⏳  Analysing…")
        self._grade_label.configure(
            text="Processing…", text_color=_TEXT_LIGHT, fg_color=_BG_PEACH,
        )

        threading.Thread(
            target=self._run_analysis,
            args=(path,),
            daemon=True,
        ).start()

    def _run_analysis(self, image_path: str) -> None:
        """Execute the detection pipeline off the main thread."""
        try:
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {image_path}")

            result = run_pipeline(bgr, model_path=_MODEL_PATH)
            severity = compute_severity(
                classifier_confidence=result.classifier_confidence,
                blemish_pixel_ratio=result.blemish_ratio,
                defect_name=result.defect_name,
                model_used=result.model_used,
            )

            self._source_bgr = bgr
            self._result = result
            self._severity = severity
            self.after(0, self._display_results)

        except Exception as exc:
            logger.exception("Pipeline failed")
            self.after(0, lambda: self._show_error(str(exc)))

    def _display_results(self) -> None:
        sev = self._severity
        res = self._result
        if sev is None or res is None:
            return

        # Disease name
        self._disease_label.configure(
            text=f"🔬  {sev.defect_name}",
            text_color=_ORANGE_DARK,
            fg_color=_BG_WARM,
        )

        # Grade badge
        self._grade_label.configure(
            text=f"  {sev.grade}  ",
            text_color=_TEXT_WHITE,
            fg_color=sev.grade_colour,
        )

        # Metric tiles
        self._stat_defect.configure(text=f"{sev.defect_pct:.1f} %")
        self._stat_conf.configure(text=f"{sev.classifier_conf * 100:.1f} %")
        self._stat_comp.configure(text=f"{sev.composite_score:.1f}")
        self._stat_model.configure(
            text="MobileNetV2" if res.model_used else "Dual-Layer",
        )

        # Treatment
        self._treatment_box.configure(state="normal")
        self._treatment_box.delete("1.0", "end")
        self._treatment_box.insert("1.0", sev.treatment)
        self._treatment_box.configure(state="disabled")

        # Preview
        self._show_preview("Original")
        self._view_toggle.set("Original")

        # Buttons
        self._upload_btn.configure(state="normal", text="📂  Upload Carrot Image")
        self._export_btn.configure(state="normal")
        self._report_btn.configure(state="normal")

    # ── Preview helpers ──────────────────────────────────────────────

    def _show_preview(self, mode: str) -> None:
        if self._source_bgr is None or self._result is None:
            return

        bgr = (
            self._source_bgr
            if mode == "Original"
            else self._result.annotated_image
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self._canvas_frame.update_idletasks()
        cw = max(self._canvas_frame.winfo_width() - 24, 200)
        ch = max(self._canvas_frame.winfo_height() - 24, 200)
        pil.thumbnail((cw, ch), Image.LANCZOS)

        self._preview_photo = ImageTk.PhotoImage(pil)
        self._canvas_label.configure(image=self._preview_photo, text="")

    def _on_view_toggle(self, value: str) -> None:
        self._show_preview(value)

    # ── Export annotated image ───────────────────────────────────────

    def _on_export(self) -> None:
        if self._result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            cv2.imwrite(path, self._result.annotated_image)
            logger.info("Annotated image saved to %s", path)

    # ── Download quality report (.txt) ───────────────────────────────

    def _on_download_report(self) -> None:
        sev = self._severity
        res = self._result
        if sev is None or res is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Quality Report",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile=f"carrot_report_{datetime.now():%Y%m%d_%H%M%S}.txt",
        )
        if not path:
            return

        report_lines = [
            "=" * 56,
            "   🥕  CARROT DEFECT DETECTION – QUALITY REPORT",
            "=" * 56,
            "",
            f"  Date / Time     : {datetime.now():%Y-%m-%d  %H:%M:%S}",
            f"  Source Image     : {self._image_path or 'N/A'}",
            f"  Pipeline         : {res.pipeline_name}",
            "",
            "-" * 56,
            "  RESULTS",
            "-" * 56,
            "",
            f"  Detected Defect  : {sev.defect_name}",
            f"  Grade            : {sev.grade}",
            f"  Defect Area      : {sev.defect_pct:.2f} %",
            f"  Confidence       : {sev.classifier_conf * 100:.2f} %",
            f"  Composite Score  : {sev.composite_score:.2f} / 100",
            "",
            "-" * 56,
            "  TREATMENT GUIDANCE",
            "-" * 56,
            "",
        ]
        for line in sev.treatment.splitlines():
            report_lines.append(f"  {line}")
        report_lines += [
            "",
            "=" * 56,
            "  Report generated by Carrot Defect Detector v1.0",
            "=" * 56,
            "",
        ]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report_lines))

        logger.info("Quality report saved to %s", path)

    # ── Error display ────────────────────────────────────────────────

    def _show_error(self, msg: str) -> None:
        self._upload_btn.configure(state="normal", text="📂  Upload Carrot Image")
        self._grade_label.configure(
            text="  Error  ", text_color=_TEXT_WHITE, fg_color=_RED_ALERT,
        )
        self._treatment_box.configure(state="normal")
        self._treatment_box.delete("1.0", "end")
        self._treatment_box.insert("1.0", f"❌  {msg}")
        self._treatment_box.configure(state="disabled")


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    app = CarrotDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
