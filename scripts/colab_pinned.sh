#!/usr/bin/env bash
# Execute the Colab notebook natively (arm64) with the Colab runtime's exact package versions.
set -euo pipefail

NB="${1:-DL4HL-Tutorial.ipynb}"
OUT="${2:-.cache/colab-pinned}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${COLAB_IMAGE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["official_colab_image"])' "$SCRIPT_DIR/../dl4hl/assets/runtime.json")}"
PKGS='numpy|pandas|scipy|scikit-learn|scikit-image|matplotlib|matplotlib-inline|seaborn|huggingface-hub|hf-xet|fsspec|pyarrow|ipython|ipykernel|jupyter_client|nbformat|nbconvert|joblib|threadpoolctl|networkx|h5py|pillow|requests|tqdm|torch|torchvision|transformers|accelerate|peft|datasets|timm|einops|safetensors|tokenizers'

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Pulling official Colab runtime image: $IMAGE"
  docker pull --platform linux/amd64 "$IMAGE"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$NB" "$WORK/notebook.ipynb"
cp "$SCRIPT_DIR/execute_notebook.py" "$WORK/execute_notebook.py"
docker image inspect "$IMAGE" > "$OUT/runtime-image.json"

echo "Reading package pins from the Colab image..."
docker run --rm --platform linux/amd64 --entrypoint /bin/bash "$IMAGE" -c 'pip freeze 2>/dev/null' \
  | grep -iE "^(${PKGS})==" | sed -E 's/\+cu[0-9]+//' > "$WORK/pins.txt"
grep -iE '^(scikit-learn|pandas|torch|transformers)==' "$WORK/pins.txt" | tr '\n' ' '; echo

echo "Executing $NB in python:3.12-slim (arm64) with those pins..."
cp "$WORK/pins.txt" "$OUT/pins.txt"
docker run --rm --platform linux/arm64 -e JOBLIB_MULTIPROCESSING=0 -e DL4HL_RUNTIME_IMAGE="native-arm64-with-pins-from-$IMAGE" -w /nb -v "$WORK:/nb:rw" -v "$OUT:/results:rw" python:3.12-slim bash -c '
apt-get update -qq >/dev/null && apt-get install -y -qq gcc g++ git >/dev/null 2>&1
pip install -q uv >/dev/null 2>&1
uv pip install --system -q -r pins.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
python execute_notebook.py notebook.ipynb --output /results/executed.ipynb'
