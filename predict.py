"""Classify a cropped face image using the model exported by LAB3/train.py."""
import argparse
from pathlib import Path

import numpy as np

from common import DISPLAY, LABELS, EXTENSIONS, image_array, load_keras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Anh da cat khuon mat hoac thu muc anh")
    parser.add_argument("--model", type=Path, default=Path("outputs/run1/mask_model.keras"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size phai duong.")
    if not args.model.is_file():
        parser.error("Chua co model. Chay train.py truoc.")
    files = sorted(p for p in args.image.iterdir() if p.suffix.lower() in EXTENSIONS) if args.image.is_dir() else [args.image]
    if not files or any(not p.is_file() for p in files):
        parser.error("Khong tim thay anh.")
    keras = load_keras()
    model = keras.models.load_model(args.model, compile=False)
    if model.output_shape[-1] != len(LABELS):
        parser.error("Model phai co 3 dau ra theo thu tu nhan LAB3.")
    shape = model.input_shape
    while isinstance(shape, list):
        shape = shape[0]
    size = int(shape[1])
    step = args.batch_size
    for start in range(0, len(files), step):
        chunk = files[start:start + step]
        batch = np.stack([image_array(p, size) for p in chunk])
        for path, probabilities in zip(chunk, model.predict(batch, verbose=0)):
            index = int(np.argmax(probabilities))
            print(f"{path.name}: {DISPLAY[index]} ({probabilities[index]:.1%})")
            for label, probability in zip(LABELS, probabilities):
                print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()
