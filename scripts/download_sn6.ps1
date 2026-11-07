# Download SpaceNet 6 (AOI 11 Rotterdam) from the public S3 bucket over HTTPS.
# Resumable: re-run the script after any interruption and curl continues where it stopped.
# Runs on the Windows side so it writes straight to D: (no WSL 9P bridge) and
# keeps going even when WSL shuts down.
#
# Usage (PowerShell):  .\scripts\download_sn6.ps1 [-Dest D:\geo\raw\sn6] [-SkipFull]

param(
    [string]$Dest = "D:\geo\raw\sn6",
    [switch]$SkipFull
)

$ErrorActionPreference = "Stop"
$base = "https://spacenet-dataset.s3.amazonaws.com"
$aoi = "AOIs/AOI_11_Rotterdam"

# Small files first so day-1 work can start, the 39 GiB tarball last.
$files = @(
    "spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train_sample.tar.gz",
    "$aoi/extents_masks/SN6_AOI_11_Rotterdam_Train-Extent.geojson",
    "$aoi/extents_masks/SN6_AOI_11_Rotterdam_TrainVal-Extent.geojson",
    "$aoi/extents_masks/SN6_AOI_11_Rotterdam_Val-Extent.geojson",
    "$aoi/extents_masks/SN6_AOI_11_Rotterdam_Test-Extent.geojson",
    "$aoi/extents_masks/SN6_AOI_11_Rotterdam_LowFidelityAnnotationMask.geojson",
    "$aoi/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_All-Train.geojson",
    "$aoi/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_All-TrainVal.geojson",
    "$aoi/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_All-Val.geojson",
    "$aoi/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-Train.geojson",
    "$aoi/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-TrainVal.geojson",
    "$aoi/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-Val.geojson",
    "$aoi/PS-RGBNIR/SN6_AOI_11_Rotterdam_PS-RGBNIR_TrainVal.tif"
)
if (-not $SkipFull) {
    $files += "spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz"
}

New-Item -ItemType Directory -Force $Dest | Out-Null
$log = Join-Path $Dest "download.log"

foreach ($key in $files) {
    $sub = if ($key -like "*geojson_buildings*") { "geojson_buildings" }
           elseif ($key -like "*extents_masks*") { "extents_masks" }
           else { "" }
    $dir = if ($sub) { Join-Path $Dest $sub } else { $Dest }
    New-Item -ItemType Directory -Force $dir | Out-Null
    $out = Join-Path $dir (Split-Path $key -Leaf)
    $url = "$base/$key"

    # Skip files that are already complete.
    $remote = [int64]((curl.exe -sI $url | Select-String -Pattern "^Content-Length:\s*(\d+)").Matches[0].Groups[1].Value)
    if ((Test-Path $out) -and ((Get-Item $out).Length -eq $remote)) {
        Add-Content $log "$(Get-Date -Format s) SKIP  $out (complete, $remote bytes)"
        continue
    }

    Add-Content $log "$(Get-Date -Format s) START $url ($remote bytes)"
    & curl.exe -L -sS -C - --retry 50 --retry-delay 15 --retry-all-errors -o $out $url 2>> $log
    $got = (Get-Item $out).Length
    $status = if ($got -eq $remote) { "DONE " } else { "SHORT" }
    Add-Content $log "$(Get-Date -Format s) $status $out ($got / $remote bytes)"
}

Add-Content $log "$(Get-Date -Format s) ALL FINISHED"
