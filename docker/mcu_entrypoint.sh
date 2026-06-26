#!/usr/bin/env bash
set -euo pipefail

role="${MCU_ROLE:-hp}"
kind="${MCU_KIND:-tensor}"
op="${MCU_OP:-elemul}"
port="${MCU_PORT:-9200}"
out_dir="${MCU_OUT_DIR:-/workspace/out/default}"
mkdir -p "${out_dir}"

if [[ "${role}" == "p0" || "${role}" == "p1" ]]; then
  sleep "${MCU_START_DELAY:-1}"
fi

if [[ "${kind}" == "tensor" ]]; then
  exe="/workspace/mcu_rust/target/release/real_tensor"
  if [[ "${op}" == "elemul" ]]; then
    shape_args=(--op elemul --len "${MCU_LEN:-64}")
  elif [[ "${op}" == "matmul" ]]; then
    shape_args=(--op matmul --m "${MCU_M:-4}" --k "${MCU_K:-16}" --n "${MCU_N:-16}")
  else
    echo "unsupported tensor op: ${op}" >&2
    exit 2
  fi
elif [[ "${kind}" == "nonlinear" ]]; then
  exe="/workspace/mcu_rust/target/release/real_nonlinear"
  shape_args=(--op "${op}" --n "${MCU_N:-64}")
  if [[ "${op}" == "softmax" ]]; then
    shape_args+=(--k "${MCU_K:-4}")
  fi
else
  echo "unsupported MCU_KIND: ${kind}" >&2
  exit 2
fi

case "${role}" in
  hp)
    exec "${exe}" hp --addr "0.0.0.0:${port}" "${shape_args[@]}"
    ;;
  p0)
    exec "${exe}" p0 --addr "mcu-hp:${port}" "${shape_args[@]}" --out "${out_dir}/p0.out"
    ;;
  p1)
    exec "${exe}" p1 --addr "mcu-hp:${port}" "${shape_args[@]}" --out "${out_dir}/p1.out"
    ;;
  verify)
    exec "${exe}" verify "${shape_args[@]}" --p0 "${out_dir}/p0.out" --p1 "${out_dir}/p1.out"
    ;;
  *)
    echo "unsupported MCU_ROLE: ${role}" >&2
    exit 2
    ;;
esac
