# Shared settings for the college-server runbook (sourced by the other scripts).
# Everything we create on the server lives under BASE, so cleanup.sh can remove it all.
# The server address is not committed (public repo): put REMOTE=user@host in
# scripts/server/remote.local (git-ignored) or export REMOTE before running.
_here="$(dirname "${BASH_SOURCE[0]}")"
[ -f "$_here/remote.local" ] && source "$_here/remote.local"
REMOTE="${REMOTE:-}"                        # needed only on the laptop side
BASE="${BASE:-/scratch/user6/geoinfer}"     # home is 87 % full; /scratch/user6 is ours (700)
REPO="$BASE/repo"
RAW="$BASE/raw"                             # downloaded tarball
DATA="$BASE/data"                           # extracted SN6 (SAR, optical, labels, summary)
ROOT="$DATA/train/AOI_11_Rotterdam"         # = data.root and --interim-root on the server
GPU="${GPU:-1}"                             # GPU 0 is used by another user
SN6_URL="https://spacenet-dataset.s3.amazonaws.com/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz"
SN6_SHA256="406e5b7763efc13f5a304b53fafd32eacab0b53e516b6a2804922fa9eef7a9e6"
SN6_SIZE=41874641730
PARTS=8                                     # parallel ranges: ~13 MB/s each from AWS
# keep uv, Python, caches and temp files inside BASE (nothing in the home directory)
SERVER_ENV="export PATH=$BASE/bin:\$PATH UV_INSTALL_DIR=$BASE/bin UV_CACHE_DIR=$BASE/cache/uv \
UV_PYTHON_INSTALL_DIR=$BASE/python XDG_CACHE_HOME=$BASE/cache MPLCONFIGDIR=$BASE/cache/mpl \
TMPDIR=$BASE/tmp"
