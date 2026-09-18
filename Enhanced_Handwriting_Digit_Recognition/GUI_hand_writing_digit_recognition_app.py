import ctypes
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 28
CANVAS_SIZE = 300
FINE = 140
SCALE = CANVAS_SIZE / FINE
MODEL_PATH = "digit_model.keras"

BG = "#1e1f22"
PANEL = "#26282e"
FG = "#e8eaf0"
MUTED = "#9aa3b2"
ACCENT = "#4f9dff"
GREEN = "#3ddc84"
AMBER = "#ffb020"
RED = "#ff5b5b"

PEN_SIZES = {0: (0.9, "S"), 1: (1.4, "M"), 2: (2.2, "L")}
DRAW, ERASE = 0, 1


def build_model():
    return Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])


def load_or_train_model(progress_cb=None, done_cb=None):
    try:
        model = load_model(MODEL_PATH)
        if progress_cb:
            progress_cb("Loaded existing model from disk.")
        if done_cb:
            done_cb(model, None)
        return model
    except Exception:
        pass

    def _train():
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

        model = build_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5),
        ]

        if progress_cb:
            progress_cb("Training for 8 epochs (first time only) ...")

        history = model.fit(X_train, y_train, epochs=8,
                            validation_data=(X_test, y_test), callbacks=callbacks)

        model.save(MODEL_PATH)
        if progress_cb:
            progress_cb("Model trained and saved to disk.")
        if done_cb:
            done_cb(model, history)

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()


def predict_digit(model, image_28):
    probs = model.predict(image_28.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence, probs


def downsample(fine):
    block = FINE // IMG_SIZE
    h, w = fine.shape[0] // block * block, fine.shape[1] // block * block
    return fine[:h].reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def center_image(img):
    yy, xx = np.nonzero(img)
    if len(yy) == 0:
        return img
    cy, cx = yy.mean(), xx.mean()
    dy = int(round(IMG_SIZE / 2 - 2 - cy))
    dx = int(round(IMG_SIZE / 2 - cx))
    out = np.zeros_like(img)
    src_y0, src_y1 = max(-dy, 0), min(IMG_SIZE, IMG_SIZE - dy)
    dst_y0, dst_y1 = max(dy, 0), min(IMG_SIZE, IMG_SIZE + dy)
    src_x0, src_x1 = max(-dx, 0), min(IMG_SIZE, IMG_SIZE - dx)
    dst_x0, dst_x1 = max(dx, 0), min(IMG_SIZE, IMG_SIZE + dx)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


class DigitDrawer(tk.Canvas):
    def __init__(self, master, on_predict, **kwargs):
        super().__init__(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#000000",
                         highlightthickness=2, highlightbackground="#3a3d45", **kwargs)
        self.on_predict = on_predict
        self.image = np.zeros((FINE, FINE), dtype=np.float32)
        self._kernels = {key: self._make_kernel(r) for key, (r, _) in PEN_SIZES.items()}
        self.pen = 1
        self.mode = DRAW
        self.eraser_on = False
        self._last = None
        self._last_predict = 0.0
        self._stroke_ids = None
        self._undo_stack = []

        self.bind("<Button-1>", self._start_stroke)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._end_stroke)

        self._draw_grid()

    @staticmethod
    def _make_kernel(radius):
        size = int(radius * 3) // 2 * 2 + 3
        y, x = np.indices((size, size))
        c = (size - 1) / 2
        g = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * radius * radius))
        return g.astype(np.float32)

    def _draw_grid(self):
        step = 10
        for i in range(1, CANVAS_SIZE // step):
            self.create_line(i * step, 0, i * step, CANVAS_SIZE, fill="#15161a")
            self.create_line(0, i * step, CANVAS_SIZE, i * step, fill="#15161a")
        for item in self.find_all():
            self.addtag_withtag("grid", item)

    def _reset_grid(self):
        self.delete("grid")
        self._draw_grid()

    # ---- stroke handling ----

    def _start_stroke(self, event):
        if self._stroke_ids is not None:
            self._finalize_stroke()
        self._stroke_ids = []
        snapshot = (self.image.copy(), self._stroke_ids)
        self._undo_stack.append(snapshot)
        self._draw_at(event)

    def _drag(self, event):
        self._draw_at(event)
        if self.on_predict:
            self.on_predict(force=False)

    def _end_stroke(self, event):
        self._finalize_stroke()
        if self.on_predict:
            self.on_predict(force=True)

    def _finalize_stroke(self):
        self._stroke_ids = None
        self._last = None

    def _draw_at(self, event):
        radius, _ = PEN_SIZES[self.pen]
        kernel = self._kernels[self.pen]
        fx = np.clip(event.x / SCALE, 0, FINE - 1)
        fy = np.clip(event.y / SCALE, 0, FINE - 1)

        if self._last is None:
            self._stamp(fx, fy, kernel, radius)
        else:
            x0, y0 = self._last
            dist = max(abs(fx - x0), abs(fy - y0))
            steps = max(int(dist / 0.6), 1)
            for i in range(1, steps + 1):
                t = i / steps
                self._stamp(x0 + (fx - x0) * t, y0 + (fy - y0) * t, kernel, radius)
        self._last = (float(fx), float(fy))
        self.tag_lower("grid")

    def _stamp(self, fx, fy, kernel, radius):
        half = (kernel.shape[0] - 1) / 2
        x0 = int(np.floor(fx - half))
        y0 = int(np.floor(fy - half))
        h, w = kernel.shape
        row0, row1 = max(y0, 0), min(y0 + h, FINE)
        col0, col1 = max(x0, 0), min(x0 + w, FINE)
        krow0, krow1 = row0 - y0, row1 - y0
        kcol0, kcol1 = col0 - x0, col1 - x0
        region = self.image[row0:row1, col0:col1]
        ke = kernel[krow0:krow1, kcol0:kcol1]
        if self.mode == DRAW:
            np.maximum(region, ke, out=region)
        else:
            np.minimum(region, 1.0 - ke, out=region)

        cx, cy = fx * SCALE, fy * SCALE
        r = max(radius * SCALE * 1.35, 1.2)
        color = "black" if self.mode == ERASE else "white"
        oid = self.create_oval(cx - r, cy - r, cx + r, cy + r,
                               fill=color, outline=color)
        if self._stroke_ids is not None:
            self._stroke_ids.append(oid)

    def clear(self):
        self.image.fill(0.0)
        self.delete("all")
        self._reset_grid()
        self._stroke_ids = None
        self._last = None
        self._undo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return False
        img, ids = self._undo_stack.pop()
        if ids:
            self.delete(*ids)
        self.image = img.copy()
        self._last = None
        self._stroke_ids = None
        return True

    def model_view(self):
        small = downsample(self.image)
        return center_image(small)


class DigitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwriting Digit Recognition")
        self.minsize(880, 600)
        self.resizable(True, True)
        self.configure(bg=BG)
        self._apply_theme()

        self.model = None
        self.history = None
        self.mnist_data = None
        self._last_tip = "Draw a digit on the left and it predicts live."

        self.status = tk.StringVar(value="Loading model ...")
        self.result_var = tk.StringVar(value="-")
        self.conf_var = tk.StringVar(value="")
        self.acc_var = tk.StringVar(value="Test accuracy: -")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._loaded = threading.Event()
        load_or_train_model(progress_cb=self._set_status, done_cb=self._on_model_ready)
        self.after(0, self._maximize)

    def _maximize(self):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")

    # ---- theme ----

    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, fieldbackground=PANEL,
                        bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL,
                        font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("TPanel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TPanel.TLabel", background=PANEL)
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Big.TLabel", font=("Segoe UI", 54, "bold"), foreground=ACCENT)
        style.configure("TButton", background="#3a3d45", foreground=FG, padding=(10, 5),
                        borderwidth=0, focuscolor=BG)
        style.map("TButton",
                  background=[("active", ACCENT), ("pressed", "#2f6fce")],
                  foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"))
        style.configure("Toolbar.TButton", padding=(8, 4))
        style.configure("TStatusbar.TLabel", foreground=MUTED, padding=(10, 4))

    # ---- ui ----

    def _build_ui(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Draw a digit", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        self.canvas = DigitDrawer(left, on_predict=self.do_predict)
        self.canvas.grid(row=1, column=0, pady=6)

        toolbar = ttk.Frame(left)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self._tooltip_button(toolbar, "Clear", "Clear the canvas", self.canvas.clear)
        self._tooltip_button(toolbar, "Undo", "Undo last stroke", self.canvas.undo)
        self.eraser_btn = self._tooltip_button(toolbar, "Eraser", "Erase strokes (toggle)",
                                               self.toggle_eraser)

        penrow = ttk.Frame(left)
        penrow.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(penrow, text="Pen:", style="Muted.TLabel").pack(side="left")
        self.pen_btns = {}
        for k, (_, name) in PEN_SIZES.items():
            b = ttk.Button(penrow, text=name[:1].upper(), width=3,
                           command=lambda key=k: self.set_pen(key))
            b.pack(side="left", padx=(3, 0))
            self.pen_btns[k] = b
        self.mode_label = tk.StringVar(value="Pen: Medium | Mode: Draw")
        ttk.Label(penrow, textvariable=self.mode_label, style="Muted.TLabel").pack(side="right")

        right = ttk.Frame(self, style="TPanel.TFrame", padding=14)
        right.pack(side="left", fill="both", expand=True)

        top = ttk.Frame(right, style="TPanel.TFrame")
        top.pack(fill="x")
        head = ttk.Frame(top, style="TPanel.TFrame")
        head.pack(side="left", fill="x", expand=True)
        ttk.Label(head, text="Prediction", style="TPanel.TLabel").pack(anchor="w")
        self.result_label = ttk.Label(head, textvariable=self.result_var, style="Big.TLabel")
        self.result_label.pack(anchor="w")
        self.conf_label = ttk.Label(head, textvariable=self.conf_var, style="TPanel.TLabel")
        self.conf_label.pack(anchor="w")
        ttk.Label(head, textvariable=self.acc_var, style="Muted.TLabel").pack(anchor="w")

        preview = ttk.Frame(top, style="TPanel.TFrame")
        preview.pack(side="right")
        self.preview = tk.Canvas(preview, width=IMG_SIZE * 5, height=IMG_SIZE * 5,
                                 bg="#000000", highlightthickness=1,
                                 highlightbackground="#3a3d45")
        self.preview.pack()
        ttk.Label(preview, text="What the model sees", style="Muted.TLabel").pack()

        ttk.Label(right, text="All-class probabilities", style="TPanel.TLabel",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        self.bars_frame = ttk.Frame(right, style="TPanel.TFrame")
        self.bars_frame.pack(fill="both", expand=True)

        more = ttk.Frame(right, style="TPanel.TFrame")
        more.pack(fill="x", pady=(10, 0))
        ttk.Button(more, text="Retrain Model", command=self.retrain).pack(side="left")
        ttk.Button(more, text="Test-Set Grid", command=self.show_test_grid).pack(side="left", padx=(8, 0))
        ttk.Button(more, text="Confusion Matrix", command=self.show_confusion).pack(side="left", padx=(8, 0))

        self.status_label = ttk.Label(self, textvariable=self.status, style="TStatusbar.TLabel")
        self.status_label.pack(side="bottom", fill="x")

        self.after(300, self._refresh_pen_ui)

    def _tooltip_button(self, parent, text, tip, command):
        btn = ttk.Button(parent, text=text, command=command, style="Toolbar.TButton")
        btn.pack(side="left", padx=(0, 6))
        btn.bind("<Enter>", lambda e: self.status.set(tip))
        btn.bind("<Leave>", lambda e: self.status.set(self._last_tip))
        return btn

    # ---- toolbar actions ----

    def toggle_eraser(self):
        self.canvas.mode = ERASE if self.canvas.mode == DRAW else DRAW
        self.canvas.eraser_on = self.canvas.mode == ERASE
        self.eraser_btn.configure(text="Drawing" if self.canvas.eraser_on else "Eraser")
        self._refresh_pen_ui()

    def set_pen(self, key):
        self.pen_btns[self.canvas.pen].configure(style="TButton")
        self.canvas.pen = key
        self.pen_btns[key].configure(style="Accent.TButton")
        self._refresh_pen_ui()

    def _refresh_pen_ui(self):
        _, name = PEN_SIZES[self.canvas.pen]
        mode = "Erase" if self.canvas.mode == ERASE else "Draw"
        self.mode_label.set(f"Pen: {name} | Mode: {mode}")
        for key, btn in self.pen_btns.items():
            btn.configure(style="Accent.TButton" if key == self.canvas.pen else "TButton")

    # ---- model lifecycle ----

    def _set_status(self, msg):
        self.after(0, lambda: self.status.set(msg))

    def _on_model_ready(self, model, history):
        self.model = model
        self.history = history
        self._loaded.set()
        self.after(0, self._notify_ready)

    def _notify_ready(self):
        self._load_mnist()
        self.status.set("Ready. Draw a digit and watch it predict live.")
        self._last_tip = "Ready. Draw a digit and watch it predict live."
        threading.Thread(target=self._compute_accuracy, daemon=True).start()

    def _compute_accuracy(self):
        try:
            self._load_mnist()
            X, y = self.mnist_data
            preds = np.argmax(self.model.predict(X[:2000], verbose=0), axis=1)
            acc = float(np.mean(preds == y[:2000]))
            self.after(0, lambda: self.acc_var.set(f"Test accuracy (2k samples): {acc * 100:.2f}%"))
        except Exception:
            pass

    def _load_mnist(self):
        if self.mnist_data is None:
            (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
            self.mnist_data = (X_test, y_test)

    def retrain(self):
        self.acc_var.set("Test accuracy: -")
        self.model = None
        self.status.set("Retraining model ...")
        load_or_train_model(progress_cb=self._set_status, done_cb=self._on_model_ready)

    def _on_close(self):
        self._loaded.set()
        try:
            plt.close('all')
        except Exception:
            pass
        self.destroy()

    # ---- prediction ----

    def do_predict(self, force=True):
        if self.model is None:
            self.status.set("Model not ready yet - please wait.")
            return
        now = time.monotonic()
        if not force and now - getattr(self, "_last_predict", 0.0) < 0.15:
            return
        self._last_predict = now
        image28 = self.canvas.model_view()
        try:
            digit, confidence, probs = predict_digit(self.model, image28)
        except Exception as e:
            self.status.set(f"Prediction error: {e}")
            return
        self.result_var.set(str(digit))
        self.conf_var.set(f"Confidence: {confidence * 100:.1f}%")
        color = GREEN if confidence >= 0.7 else AMBER if confidence >= 0.5 else RED
        self.result_label.configure(foreground=color)
        self.conf_label.configure(foreground=color)
        self._draw_probability_bars(probs)
        self._draw_preview(image28)
        if force and confidence < 0.5:
            tip = "Low confidence — make the digit bigger / thicker and keep it centered."
            self.status.set(tip)
            self._last_tip = tip

    def _draw_probability_bars(self, probs):
        for child in self.bars_frame.winfo_children():
            child.destroy()
        max_prob = probs.max() or 1.0
        for i, p in enumerate(probs):
            row = ttk.Frame(self.bars_frame, style="TPanel.TFrame")
            row.pack(fill="x", pady=1)
            bg = "#2c2f37"
            ttk.Label(row, text=f"  {i}", background=bg, foreground=FG,
                      width=3, anchor="center").pack(side="left", padx=(0, 6))
            bar = tk.Canvas(row, width=280, height=11, bg="#2c2f37",
                            highlightthickness=0, bd=0)
            bar.pack(side="left", padx=(0, 8))
            fill = int(p / max_prob * 274)
            color = GREEN if i == int(np.argmax(probs)) else ACCENT
            bar.create_rectangle(0, 0, fill, 11, fill=color, outline="")
            lab = tk.Label(row, text=f"{p * 100:5.1f}%", bg=PANEL, fg=FG,
                           font=("Segoe UI", 9))
            lab.pack(side="left")

    def _draw_preview(self, image28):
        self.preview.delete("all")
        scale = 5
        for yy in range(IMG_SIZE):
            for xx in range(IMG_SIZE):
                v = float(image28[yy, xx])
                if v > 0.05:
                    g = int(255 * min(v, 1.0))
                    self.preview.create_rectangle(
                        xx * scale, yy * scale,
                        (xx + 1) * scale, (yy + 1) * scale,
                        fill=f"#{g:02x}{g:02x}{g:02x}", outline="")

    # ---- matplotlib views ----

    def show_test_grid(self):
        if self.model is None:
            return
        self._load_mnist()
        X_test, y_test = self.mnist_data
        preds = np.argmax(self.model.predict(X_test[:18], verbose=0), axis=1)

        fig, axes = plt.subplots(2, 9, figsize=(14, 3.5))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(X_test[idx, :, :, 0], cmap="gray")
            ok = preds[idx] == y_test[idx]
            ax.set_title(f"P:{preds[idx]} / T:{y_test[idx]}",
                         color="green" if ok else "red", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        plt.show(block=False)
        fig.canvas.draw_idle()

    def show_confusion(self):
        if self.model is None:
            return
        self._load_mnist()
        X_test, y_test = self.mnist_data
        preds = np.argmax(self.model.predict(X_test[:2000], verbose=0), axis=1)
        cm = confusion_matrix(y_test[:2000], preds)
        accuracy = np.trace(cm) / cm.sum()

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cm, cmap="Blues")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (2000 samples) - Accuracy: {accuracy * 100:.2f}%")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        fig.tight_layout()
        plt.show(block=False)
        fig.canvas.draw_idle()


def confusion_matrix(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = DigitApp()
    app.mainloop()


if __name__ == "__main__":
    main()