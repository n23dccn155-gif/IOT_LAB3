"""Train a lightweight three-class MobileNetV2 and evaluate held-out test data."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from common import LABELS, EXTENSIONS, image_array, load_keras


def scan_split(folder):
    samples = []
    for index, label in enumerate(LABELS):
        files = sorted(p for p in (folder / label).glob("*") if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"Thieu anh cho nhan {label}: {folder}. Chay prepare_dataset.py truoc.")
        samples.extend((p, index) for p in files)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("outputs/run1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, choices=[96, 128, 160, 192, 224], default=224)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.fine_tune_epochs < 0:
        parser.error("Epochs, batch size phai duong; fine-tune epochs khong am.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Thu muc output khong trong. Dung --out outputs/run2 de giu ket qua cu.")
    try:
        samples = {s: scan_split(args.data / s) for s in ["train", "val", "test"]}
    except ValueError as error:
        parser.exit(1, f"Loi: {error}\n")
    keras = load_keras()
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    keras.utils.set_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Backend: {keras.backend.backend()}, device: {device}")
    if args.weights == "none":
        print("CANH BAO: weights=none chi de kiem tra pipeline; khong phai transfer learning.")

    class Batches(keras.utils.PyDataset):
        def __init__(self, rows, augment=False):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.rows = rows
            self.augment = augment
            self.rng = np.random.default_rng(args.seed)
            self.order = np.arange(len(rows))
            if augment:
                self.rng.shuffle(self.order)

        def __len__(self):
            return math.ceil(len(self.rows) / args.batch_size)

        def __getitem__(self, index):
            indices = self.order[index * args.batch_size:(index + 1) * args.batch_size]
            rows = [self.rows[i] for i in indices]
            x = np.stack([image_array(p, args.image_size, self.rng if self.augment else None) for p, _ in rows])
            return x, np.array([y for _, y in rows], dtype=np.int64)

        def on_epoch_end(self):
            if self.augment:
                self.rng.shuffle(self.order)

    class MacroF1(keras.metrics.Metric):
        # Accuracy alone hides a rare class: mask_weared_incorrect has ~2% of samples.
        def __init__(self, **kwargs):
            super().__init__(name="macro_f1", **kwargs)
            shape = (len(LABELS),)
            self.tp = self.add_weight(name="tp", shape=shape, initializer="zeros")
            self.fp = self.add_weight(name="fp", shape=shape, initializer="zeros")
            self.fn = self.add_weight(name="fn", shape=shape, initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            truth = keras.ops.argmax(y_true, axis=-1)
            guess = keras.ops.argmax(y_pred, axis=-1)
            for index in range(len(LABELS)):
                hit = keras.ops.cast(keras.ops.logical_and(truth == index, guess == index), "float32")
                real = keras.ops.cast(truth == index, "float32")
                seen = keras.ops.cast(guess == index, "float32")
                self.tp.assign_add(keras.ops.sum(hit))
                self.fp.assign_add(keras.ops.sum(seen - hit))
                self.fn.assign_add(keras.ops.sum(real - hit))

        def result(self):
            precision = self.tp / (self.tp + self.fp + 1e-9)
            recall = self.tp / (self.tp + self.fn + 1e-9)
            return keras.ops.mean(2 * precision * recall / (precision + recall + 1e-9))

        def reset_state(self):
            for weight in (self.tp, self.fp, self.fn):
                weight.assign(keras.ops.zeros_like(weight))

    train, val, test = (Batches(samples[s], augment=s == "train") for s in ["train", "val", "test"])
    base = keras.applications.MobileNetV2(input_shape=(args.image_size, args.image_size, 3),
                                         alpha=0.35, include_top=False,
                                         weights="imagenet" if args.weights == "imagenet" else None)
    base.trainable = False
    inputs = keras.Input(shape=(args.image_size, args.image_size, 3), name="rgb_pixels_0_255")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalize")(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax", name="mask_classes")(x)
    model = keras.Model(inputs, outputs, name="lab3_mask_mobilenetv2")
    args.out.mkdir(parents=True, exist_ok=True)
    counts = np.bincount([y for _, y in samples["train"]], minlength=len(LABELS))
    class_weight = {i: len(samples["train"]) / (len(LABELS) * int(n)) for i, n in enumerate(counts)}
    history = {}
    best_path = args.out / "mask_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True)

    def fit_stage(epochs, learning_rate):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy", MacroF1()])
        result = model.fit(train, validation_data=val, epochs=epochs, shuffle=False,
                           class_weight=class_weight, callbacks=[checkpoint,
                           keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                           keras.callbacks.CSVLogger(str(args.out / "training_log.csv"), append=bool(history))])
        for name, values in result.history.items():
            history.setdefault(name, []).extend(float(v) for v in values)

    fit_stage(args.epochs, 1e-3)
    head_epochs = len(history["loss"])
    if args.fine_tune_epochs:
        # Begin from the best head-training checkpoint; keep BN statistics frozen.
        model.set_weights(keras.models.load_model(best_path, compile=False).get_weights())
        base.trainable = True
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-30:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
        fit_stage(args.fine_tune_epochs, 1e-5)

    best = keras.models.load_model(best_path, compile=False)
    probabilities = best.predict(test, verbose=0)
    truth = np.array([y for _, y in samples["test"]])
    predicted = probabilities.argmax(axis=1)
    report = classification_report(truth, predicted, labels=list(range(len(LABELS))),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(LABELS))))
    versions = {n: importlib.metadata.version(n) for n in ["keras", "torch", "numpy", "pillow", "scikit-learn"]}
    best_epoch = int(np.argmin(history["val_loss"]))
    metadata = {"labels": LABELS, "image_size": args.image_size, "input_range": [0, 255],
                "normalization": "Inside model: pixels / 127.5 - 1", "face_padding": 0.18,
                "architecture": "MobileNetV2 alpha=0.35", "weights": args.weights,
                "seed": args.seed, "device": device, "versions": versions, "class_weight": class_weight,
                "head_epochs_completed": head_epochs, "fine_tune_epochs_completed": len(history["loss"]) - head_epochs,
                "samples": {s: len(rows) for s, rows in samples.items()}, "test_metrics": report,
                "selected_epoch": best_epoch, "epochs_run": len(history["loss"]),
                "validation_at_selected_epoch": {k: v[best_epoch] for k, v in history.items()
                                                 if k.startswith("val_")},
                "test_macro_f1": report["macro avg"]["f1-score"],
                "per_class_f1": {label: report[label]["f1-score"] for label in LABELS}}
    if (args.data / "summary.json").is_file():
        preparation = json.loads((args.data / "summary.json").read_text(encoding="utf-8"))
        metadata["dataset_preparation"] = preparation
        metadata["face_padding"] = preparation["padding"]
    for name, value in [("metadata.json", metadata), ("classification_report.json", report), ("history.json", history)]:
        (args.out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    (args.out / "classification_report.txt").write_text(classification_report(
        truth, predicted, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0), encoding="utf-8")
    log_path = args.out / "training_log.csv"
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savetxt(args.out / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d", header=",".join(LABELS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "accuracy", "macro_f1"]):
        ax.plot(history[metric], label="train")
        ax.plot(history["val_" + metric], label="val")
        if args.fine_tune_epochs:
            ax.axvline(head_epochs - 0.5, color="gray", linestyle="--", label="fine-tune")
        ax.set(xlabel="Epoch (0-based)", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "training_curves.png")
    plt.close(fig)
    print((args.out / "classification_report.txt").read_text(encoding="utf-8"))
    print(f"Model va bao cao: {args.out.resolve()}")


if __name__ == "__main__":
    main()
