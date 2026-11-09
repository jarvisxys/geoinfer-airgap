#!/usr/bin/env bash
# On the server: download the SN6 train tarball from AWS in PARTS parallel byte ranges
# (one connection gets ~13 MB/s, eight about 50 MB/s), join, verify the SHA-256 recorded
# on the laptop, and extract only what training needs.
# Usage: remote_data.sh start   (launch the range downloads, detached)
#        remote_data.sh finish  (after every part.N.ok exists: join, verify, extract)
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$RAW" "$DATA"
cd "$RAW"
tarball="$(basename "$SN6_URL")"
case "${1:-}" in
start)
    ch=$(( (SN6_SIZE + PARTS - 1) / PARTS ))
    : > ranges.txt
    for i in $(seq 0 $((PARTS - 1))); do
        s=$((i * ch)); e=$(( (i + 1) * ch - 1 )); [ $e -ge $SN6_SIZE ] && e=$((SN6_SIZE - 1))
        echo "$i $s $e" >> ranges.txt
        setsid nohup bash -c "until curl -sS --fail -r $s-$e -C - -o part.$i $SN6_URL; \
do sleep 3; done; echo done > part.$i.ok" > "part.$i.log" 2>&1 < /dev/null &
    done
    echo "started $PARTS range downloads in $RAW"
    ;;
finish)
    if [ ! -f "$tarball" ]; then
        n_ok=$(ls part.*.ok 2>/dev/null | wc -l)
        [ "$n_ok" -eq "$PARTS" ] || { echo "only $n_ok of $PARTS parts done"; exit 1; }
        cat $(for i in $(seq 0 $((PARTS - 1))); do echo part.$i; done) > "$tarball"
    fi
    [ "$(stat -c %s "$tarball")" -eq "$SN6_SIZE" ] || { echo "size mismatch"; exit 1; }
    echo "$SN6_SHA256  $tarball" | sha256sum -c -
    rm -f part.*
    unzip_cmd="gzip -dc"
    command -v pigz >/dev/null && unzip_cmd="pigz -dc"
    # member names start with ./ (checked with tar -t)
    $unzip_cmd "$tarball" | tar -x -C "$DATA" --wildcards \
        './train/AOI_11_Rotterdam/SAR-Intensity/*' './train/AOI_11_Rotterdam/PS-RGBNIR/*' \
        './train/AOI_11_Rotterdam/geojson_buildings/*' './train/AOI_11_Rotterdam/SummaryData/*'
    for d in SAR-Intensity PS-RGBNIR geojson_buildings; do
        echo "$d: $(ls "$ROOT/$d" | wc -l) files"
    done
    ;;
*)
    echo "usage: $0 start|finish"; exit 2
    ;;
esac
