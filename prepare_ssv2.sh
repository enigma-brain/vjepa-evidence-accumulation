#!/bin/bash
# Download Something-Something-v2 (public HF mirror of the original Qualcomm archives) and unpack.
# Usage: sbatch prepare_ssv2.sh     (or: bash prepare_ssv2.sh)
#SBATCH --job-name=ssv2-download
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

ROOT=${EVACC_ROOT:-/scratch/users/liubr/aa228}
DST=$ROOT/data/ssv2
mkdir -p "$DST/raw" "$DST/videos"

eval "$(micromamba shell hook --shell bash)"
micromamba activate ${EVACC_ENV:-evacc}

# 1. Download labels, splits and the 20 archive parts (~19.4 GB).
hf download morpheushoc/something-something-v2 --repo-type dataset --local-dir "$DST/raw"
cp "$DST/raw"/{labels.json,train.json,validation.json,test.json} "$DST/"

# 2. The parts are pieces of one tar.gz: concatenate and extract -> 20bn-something-something-v2/<id>.webm
cd "$DST"
cat raw/videos/20bn-something-something-v2-?? | tar -xz
rmdir videos && mv 20bn-something-something-v2 videos   # (mv dir/* fails: 220k args)
echo "videos: $(ls videos | wc -l)"   # expect 220847
