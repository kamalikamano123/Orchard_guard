from pathlib import Path
import shutil, random, math

SOURCE = Path("../data").resolve()     # raw class folders
OUT    = Path("../dataset").resolve()  # target split folders
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
random.seed(42)

def ensure_empty(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def main():
    classes = [d for d in SOURCE.iterdir() if d.is_dir()]
    for s in SPLITS:
        ensure_empty(OUT / s)

    for c in classes:
        imgs = [p for p in c.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = math.floor(n*SPLITS["train"])
        n_val   = math.floor(n*SPLITS["val"])
        splits = {
            "train": imgs[:n_train],
            "val":   imgs[n_train:n_train+n_val],
            "test":  imgs[n_train+n_val:]
        }
        for s, items in splits.items():
            dest = OUT/s/c.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in items:
                shutil.copy2(img, dest/img.name)

    print("✅ Done → backend/dataset/{train,val,test}")

if __name__ == "__main__":
    main()
