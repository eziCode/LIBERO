#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Download one small TaF-Dataset sequence for time-frequency analysis.

Usage:
  scripts/download_taf_subset.sh [options]

Options:
  --sequence PATH  Sequence relative to taf_dataset/
                   (default: gs_mini/gs_mini_obj1)
  --output DIR     Download root (default: datasets/taf_subset)
  --with-video     Also download the tactile MP4 (~123 MB for the default)
  --token TOKEN    Hugging Face token, if required
  -h, --help       Show this help

The default force/pressure parquet and metadata total approximately 2.2 MB.
EOF
}

repo_id="jiamig/taf-dataset"
sequence="gs_mini/gs_mini_obj1"
output_dir="datasets/taf_subset"
token="${HF_TOKEN:-}"
with_video=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence)
            sequence="${2:?--sequence requires a value}"
            shift 2
            ;;
        --output)
            output_dir="${2:?--output requires a value}"
            shift 2
            ;;
        --with-video)
            with_video=1
            shift
            ;;
        --token)
            token="${2:?--token requires a value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Keep the path scoped to a sequence inside taf_dataset.
if [[ "$sequence" = /* || "$sequence" == *".."* ]]; then
    echo "--sequence must be a relative path without '..'." >&2
    exit 2
fi

if command -v hf >/dev/null 2>&1; then
    hf_cmd=(hf download "$repo_id" --repo-type dataset --local-dir "$output_dir")
elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_cmd=(huggingface-cli download "$repo_id" --repo-type dataset --local-dir "$output_dir")
else
    echo "Install the Hugging Face CLI first: python -m pip install -U huggingface_hub" >&2
    exit 1
fi

base="taf_dataset/$sequence"
files=(
    "$base/data/chunk-000/file-000.parquet"
    "$base/meta/episodes/chunk-000/file-000.parquet"
    "$base/meta/info.json"
    "$base/meta/stats.json"
    "$base/meta/tasks.parquet"
)

if [[ "$with_video" -eq 1 ]]; then
    files+=("$base/videos/observation.image/chunk-000/file-000.mp4")
fi

hf_cmd+=("${files[@]}")
if [[ -n "$token" ]]; then
    hf_cmd+=(--token "$token")
fi

"${hf_cmd[@]}"

echo "Downloaded TaF subset to: $output_dir/$base"
if [[ "$with_video" -eq 0 ]]; then
    echo "Tactile video was skipped; rerun with --with-video if needed."
fi
