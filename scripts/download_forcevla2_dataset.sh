#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Download the ForceVLA2 dataset from a verified release location.

The official project page currently says the dataset is "coming soon", so the
release location must be supplied explicitly once it is published.

Usage:
  scripts/download_forcevla2_dataset.sh --hf-repo OWNER/REPO [options]
  scripts/download_forcevla2_dataset.sh --url ARCHIVE_URL [options]

Source options (choose exactly one):
  --hf-repo ID       Hugging Face dataset repository ID
  --url URL          Direct URL to a .zip, .tar, .tar.gz, or .tgz archive

Other options:
  --output DIR       Destination directory (default: datasets/forcevla2)
  --revision REV     Hugging Face revision (default: main)
  --token TOKEN      Hugging Face token for a gated/private release
  --no-extract       Keep a direct-download archive without extracting it
  -h, --help         Show this help

The download is resumable. HF_TOKEN is also honored when --token is omitted.
EOF
}

hf_repo=""
archive_url=""
output_dir="datasets/forcevla2"
revision="main"
token="${HF_TOKEN:-}"
extract_archive=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-repo)
            hf_repo="${2:?--hf-repo requires a value}"
            shift 2
            ;;
        --url)
            archive_url="${2:?--url requires a value}"
            shift 2
            ;;
        --output)
            output_dir="${2:?--output requires a value}"
            shift 2
            ;;
        --revision)
            revision="${2:?--revision requires a value}"
            shift 2
            ;;
        --token)
            token="${2:?--token requires a value}"
            shift 2
            ;;
        --no-extract)
            extract_archive=0
            shift
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

if [[ -n "$hf_repo" && -n "$archive_url" ]] || [[ -z "$hf_repo" && -z "$archive_url" ]]; then
    echo "Supply exactly one of --hf-repo or --url." >&2
    exit 2
fi

mkdir -p "$output_dir"

if [[ -n "$hf_repo" ]]; then
    if command -v hf >/dev/null 2>&1; then
        hf_cmd=(hf download "$hf_repo" --repo-type dataset --revision "$revision" --local-dir "$output_dir")
    elif command -v huggingface-cli >/dev/null 2>&1; then
        hf_cmd=(huggingface-cli download "$hf_repo" --repo-type dataset --revision "$revision" --local-dir "$output_dir")
    else
        echo "Install the Hugging Face CLI first: python -m pip install -U huggingface_hub" >&2
        exit 1
    fi

    if [[ -n "$token" ]]; then
        hf_cmd+=(--token "$token")
    fi

    "${hf_cmd[@]}"
    echo "ForceVLA2 dataset downloaded to: $output_dir"
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for direct archive downloads." >&2
    exit 1
fi

url_path="${archive_url%%\?*}"
archive_name="${url_path##*/}"
if [[ -z "$archive_name" ]]; then
    archive_name="forcevla2_dataset.archive"
fi
archive_path="$output_dir/$archive_name"

curl --fail --location --retry 5 --continue-at - --output "$archive_path" "$archive_url"

if [[ "$extract_archive" -eq 0 ]]; then
    echo "ForceVLA2 archive downloaded to: $archive_path"
    exit 0
fi

case "$archive_name" in
    *.tar.gz|*.tgz)
        tar -xzf "$archive_path" -C "$output_dir"
        ;;
    *.tar)
        tar -xf "$archive_path" -C "$output_dir"
        ;;
    *.zip)
        if ! command -v unzip >/dev/null 2>&1; then
            echo "unzip is required to extract $archive_name." >&2
            exit 1
        fi
        unzip -n "$archive_path" -d "$output_dir"
        ;;
    *)
        echo "Downloaded $archive_path but did not recognize its archive type; leaving it unextracted." >&2
        exit 0
        ;;
esac

echo "ForceVLA2 dataset downloaded and extracted to: $output_dir"
