#!/usr/bin/env bash
# Export to ONNX (cloud) then build the TensorRT engine (on the Jetson).

set -euo pipefail

STAGE="${1:-export}"

if [ "$STAGE" = "export" ]; then
  CKPT="${2:?usage: $0 export /workspace/results/A_synth/train/xxx.pth}"
  OUT="${3:-/workspace/results/export/rtdetr.onnx}"
  mkdir -p "$(dirname "$OUT")"

  rtdetr export \
    -e /workspace/specs/rtdetr_A_synth.yaml \
    export.checkpoint="$CKPT" \
    export.onnx_file="$OUT" \
    export.batch_size=1 \
    export.opset_version=17 \
    export.on_cpu=False

  python - <<PY
import onnx
m = onnx.load("$OUT")
onnx.checker.check_model(m)
print("inputs :", [(i.name, [d.dim_value or d.dim_param for d in
      i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("outputs:", [o.name for o in m.graph.output])
PY
  exit 0
fi

if [ "$STAGE" = "engine" ]; then
  ONNX="${2:?usage: $0 engine ~/rtdetr.onnx}"
  ENGINE="${ONNX%.onnx}_fp16.engine"

  sudo nvpmodel -q || true         
  sudo jetson_clocks || true       

  /usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    --memPoolSize=workspace:2048 \
    --verbose

  echo "Benchmark"
  /usr/src/tensorrt/bin/trtexec \
    --loadEngine="$ENGINE" \
    --iterations=200 --avgRuns=100 --warmUp=1000 --useSpinWait
  exit 0
fi

exit 1
