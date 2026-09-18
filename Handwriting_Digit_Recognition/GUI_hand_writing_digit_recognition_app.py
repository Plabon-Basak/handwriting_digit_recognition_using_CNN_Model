import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 28
CANVAS_SIZE = 280
CELL = CANVAS_SIZE // IMG_SIZE
MODEL_PATH = "digit_model.keras"


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


def confusion_matrix(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


class DigitDrawer(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black",
                         highlightthickness=1, highlightbackground="gray", **kwargs)
        self.image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        self._last = None
        self.bind("<B1-Motion>", self._draw)
        self.bind("<Button-1>", self._draw)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, '_last', None))

    def _draw(self, event):
        x = min(max(event.x // CELL, 0), IMG_SIZE - 1)
        y = min(max(event.y // CELL, 0), IMG_SIZE - 1)
        if self._last:
            self._line(self._last, (x, y))
        else:
            self._paint_cell(x, y, 1.0)
        self._last = (x, y)

    def _line(self, start, end):
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0)) or 1
        for i in range(steps + 1):
            t = i / steps
            self._paint_cell(int(round(x0 + (x1 - x0) * t)),
                             int(round(y0 + (y1 - y0) * t)), 1.0)

    def _paint_cell(self, x, y, value):
        self.image[y, x] = value
        self.create_oval(x * CELL + 1, y * CELL + 1,
                         (x + 1) * CELL - 1, (y + 1) * CELL - 1,
                         fill="white", outline="white")

    def clear(self):
        self.image.fill(0.0)
        self.delete("all")


class DigitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwriting Digit Recognition")
        self.geometry("780x620")
        self.minsize(680, 580)
        self.resizable(True, True)

        self.model = None
        self.history = None
        self.mnist_data = None

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")

        self.status = tk.StringVar(value="Loading model ...")
        self.result_var = tk.StringVar(value="Prediction: -")
        self.conf_var = tk.StringVar(value="Confidence: -")
        self.probs_var = tk.StringVar(value="")

        ttk.Label(left, text="Draw a digit below:", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = DigitDrawer(left)
        self.canvas.grid(row=1, column=0, pady=6)

        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, pady=(4, 0))
        ttk.Button(btns, text="Clear", command=self.canvas.clear).pack(side="left")
        ttk.Button(btns, text="Predict", command=self.do_predict).pack(side="left", padx=(8, 0))

        right = ttk.Frame(self, padding=10)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Result", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.result_label = ttk.Label(right, textvariable=self.result_var, font=("Segoe UI", 28, "bold"))
        self.result_label.pack(anchor="w", pady=(6, 0))
        self.conf_label = ttk.Label(right, textvariable=self.conf_var, font=("Segoe UI", 12))
        self.conf_label.pack(anchor="w")

        ttk.Label(right, text="All-class probabilities:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 2))
        self.bars_frame = ttk.Frame(right)
        self.bars_frame.pack(fill="both", expand=True)

        more = ttk.Frame(right)
        more.pack(fill="x", pady=(8, 0))
        ttk.Button(more, text="Retrain Model", command=self.retrain).pack(side="left")
        ttk.Button(more, text="Test-Set Grid", command=self.show_test_grid).pack(side="left", padx=(8, 0))
        ttk.Button(more, text="Confusion Matrix", command=self.show_confusion).pack(side="left", padx=(8, 0))

        self.status_label = ttk.Label(self, textvariable=self.status, foreground="gray", padding=(10, 4))
        self.status_label.pack(side="bottom", fill="x")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._loaded = threading.Event()
        load_or_train_model(progress_cb=self._set_status, done_cb=self._on_model_ready)

    # ---- model lifecycle ----

    def _set_status(self, msg):
        self.after(0, lambda: self.status.set(msg))

    def _on_model_ready(self, model, history):
        self.model = model
        self.history = history
        self._loaded.set()
        self.after(0, self._notify_ready)

    def _notify_ready(self):
        self.status.set("Ready. Draw a digit and press Predict.")
        if self.model is not None:
            self._load_mnist()

    def _load_mnist(self):
        if self.mnist_data is None:
            (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
            self.mnist_data = (X_test, y_test)

    def retrain(self):
        if self.model is not None:
            self.model = None
            load_or_train_model(progress_cb=self._set_status, done_cb=self._on_model_ready)

    def _on_close(self):
        self._loaded.set()
        try:
            plt.close('all')
        except Exception:
            pass
        self.destroy()

    # ---- prediction ----

    def do_predict(self):
        if self.model is None:
            self.status.set("Model not ready yet - please wait.")
            return
        digit, confidence, probs = predict_digit(self.model, self.canvas.image)
        self.result_var.set(f"Prediction: {digit}")
        self.conf_var.set(f"Confidence: {confidence * 100:.1f}%")
        self._draw_probability_bars(probs)

    def _draw_probability_bars(self, probs):
        for child in self.bars_frame.winfo_children():
            child.destroy()
        max_prob = probs.max() or 1.0
        for i, p in enumerate(probs):
            row = ttk.Frame(self.bars_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{i}", width=2).pack(side="left")
            bar = tk.Canvas(row, width=200, height=10, bg="lightgray", highlightthickness=0)
            bar.pack(side="left", padx=(0, 6))
            fill = int(p / max_prob * 194)
            color = "green" if i == int(np.argmax(probs)) else "#7a7aff"
            bar.create_rectangle(0, 0, fill, 10, fill=color, outline="")
            ttk.Label(row, text=f"{p * 100:.1f}%", width=6).pack(side="left")

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
            ax.set_title(f"P:{preds[idx]} / T:{y_test[idx]}", color="green" if ok else "red",
                         fontsize=9)
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


if __name__ == "__main__":
    app = DigitApp()
    app.mainloop()