#!/usr/bin/env python3
"""
USGS 1 Arc-Second DEM Downloader and Tile Generator

Downloads USGS 1 arc-second elevation data for US states,
combines TIFFs using GDAL, and generates 512x512 tiles.

USGS 1 arc-second (~30m resolution) National Elevation Dataset (NED)
is accessed via The National Map (TNM) API.
"""

import sys
import time
import argparse
import subprocess
import zipfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# State bounding boxes (minLon, minLat, maxLon, maxLat)
STATE_BOUNDS = {
    "AL": (-88.473, 30.223, -84.889, 35.008),
    "AK": (-179.148, 51.214, -129.980, 71.365),
    "AZ": (-114.814, 31.332, -109.045, 37.004),
    "AR": (-94.618, 33.004, -89.644, 36.500),
    "CA": (-124.409, 32.534, -114.131, 42.009),
    "CO": (-109.060, 36.992, -102.041, 41.003),
    "CT": (-73.728, 40.987, -71.787, 42.050),
    "DE": (-75.788, 38.451, -75.049, 39.839),
    "FL": (-87.635, 24.523, -80.031, 31.001),
    "GA": (-85.605, 30.357, -80.840, 35.001),
    "HI": (-160.074, 18.948, -154.807, 22.235),
    "ID": (-117.243, 41.988, -111.044, 49.001),
    "IL": (-91.513, 36.970, -87.020, 42.508),
    "IN": (-88.098, 37.772, -84.784, 41.761),
    "IA": (-96.639, 40.375, -90.140, 43.501),
    "KS": (-102.052, 36.993, -94.588, 40.003),
    "KY": (-89.571, 36.497, -81.965, 39.147),
    "LA": (-94.043, 28.928, -88.817, 33.019),
    "ME": (-71.084, 43.064, -66.950, 47.460),
    "MD": (-79.487, 37.912, -75.049, 39.723),
    "MA": (-73.508, 41.238, -69.929, 42.887),
    "MI": (-90.418, 41.696, -82.122, 48.190),
    "MN": (-97.239, 43.499, -89.491, 49.384),
    "MS": (-91.655, 30.174, -88.098, 34.996),
    "MO": (-95.774, 35.995, -89.099, 40.613),
    "MT": (-116.050, 44.358, -104.039, 49.001),
    "NE": (-104.053, 39.999, -95.308, 43.001),
    "NV": (-120.006, 35.002, -114.040, 42.002),
    "NH": (-72.557, 42.697, -70.703, 45.305),
    "NJ": (-75.563, 38.929, -73.894, 41.357),
    "NM": (-109.050, 31.332, -103.002, 37.000),
    "NY": (-79.762, 40.496, -71.856, 45.016),
    "NC": (-84.322, 33.842, -75.460, 36.588),
    "ND": (-104.049, 45.935, -96.554, 49.001),
    "OH": (-84.820, 38.403, -80.519, 41.978),
    "OK": (-103.002, 33.616, -94.431, 37.002),
    "OR": (-124.567, 41.992, -116.463, 46.292),
    "PA": (-80.519, 39.720, -74.690, 42.270),
    "RI": (-71.862, 41.147, -71.120, 42.019),
    "SC": (-83.354, 32.035, -78.541, 35.215),
    "SD": (-104.058, 42.480, -96.436, 45.945),
    "TN": (-90.310, 34.983, -81.647, 36.678),
    "TX": (-106.646, 25.837, -93.508, 36.500),
    "UT": (-114.053, 36.998, -109.041, 42.001),
    "VT": (-73.438, 42.727, -71.465, 45.017),
    "VA": (-83.675, 36.541, -75.242, 39.466),
    "WA": (-124.849, 45.544, -116.916, 49.002),
    "WV": (-82.644, 37.202, -77.719, 40.638),
    "WI": (-92.889, 42.492, -86.806, 47.080),
    "WY": (-111.056, 40.995, -104.052, 45.006),
    "DC": (-77.119, 38.792, -76.909, 38.996),
    "PR": (-67.945, 17.881, -65.221, 18.516),
}

# Region bounding boxes (minLon, minLat, maxLon, maxLat)
REGION_BOUNDS = {
    "AK":  (-180, 51, -126, 71),   # Alaska
    "PAC": (-162, 18, -152, 24),   # Pacific (Hawaii)
    "NW":  (-125, 40, -103, 50),   # Northwest
    "SW":  (-125, 15, -103, 40),   # Southwest
    "NC":  (-105, 37, -90, 50),    # North Central
    "EC":  (-95, 37, -80, 50),     # East Central
    "SC":  (-110, 15, -90, 37),    # South Central
    "NE":  (-80, 37, -60, 50),     # Northeast
    "SE":  (-90, 15, -60, 37),     # Southeast
}

# TNM API endpoint for searching datasets
TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"


def query_api_with_retry(params: dict, max_retries: int = 3) -> list:
    """Query TNM API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(TNM_API_URL, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
        except requests.RequestException as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
    return None


def get_dem_products(
    name: str,
    bounds: tuple = None,
    resolution: str = "1 arc-second"
) -> list:
    """
    Query TNM API for 1 arc-second DEM products within bounds.
    Splits large areas into smaller sub-queries to avoid timeouts.
    
    Args:
        name: Name for logging (state code or region name)
        bounds: Tuple of (minLon, minLat, maxLon, maxLat). If None, looks up from STATE_BOUNDS.
        resolution: Resolution string (default "1 arc-second")
        
    Returns:
        List of product dictionaries with download URLs
    """
    import re
    
    if bounds is None:
        if name not in STATE_BOUNDS:
            raise ValueError(f"Unknown state code: {name}")
        bounds = STATE_BOUNDS[name]
    
    min_lon, min_lat, max_lon, max_lat = bounds
    
    print(f"Querying TNM API for {name} DEM products...")
    
    # Split large bounding boxes into smaller 5x5 degree tiles
    tile_size = 5
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    sub_boxes = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            sub_boxes.append((
                lon,
                lat,
                min(lon + tile_size, max_lon),
                min(lat + tile_size, max_lat)
            ))
            lat += tile_size
        lon += tile_size
    
    if len(sub_boxes) > 1:
        print(f"  Splitting into {len(sub_boxes)} sub-queries to avoid timeouts")
    
    all_products = []
    
    for i, (smin_lon, smin_lat, smax_lon, smax_lat) in enumerate(sub_boxes):
        if len(sub_boxes) > 1:
            print(f"  Querying sub-region {i+1}/{len(sub_boxes)}: ({smin_lon},{smin_lat}) to ({smax_lon},{smax_lat})")
        
        offset = 0
        page_size = 500
        
        while True:
            # Query Elevation Products (includes NED, 3DEP, etc.)
            params = {
                "prodFormats": "GeoTIFF",
                "prodExtents": "1 x 1 degree",
                "bbox": f"{smin_lon},{smin_lat},{smax_lon},{smax_lat}",
                "outputFormat": "JSON",
                "max": page_size,
                "offset": offset,
            }
            
            products = query_api_with_retry(params)
            
            if products is None:
                print(f"    Failed to query sub-region, skipping")
                break
            
            if not products:
                break
            
            # Filter for 1 arc-second elevation TIFFs
            for p in products:
                url = p.get("downloadURL", "")
                title = p.get("title", "").lower()
                
                # Include 1 arc-second products (NED or 3DEP)
                if url.endswith(".tif"):
                    if "1 arc-second" in title or "USGS_1_" in url or "/1/" in url:
                        all_products.append(p)
            
            if len(products) < page_size:
                break
            
            offset += page_size
            time.sleep(0.5)
        
        time.sleep(1)  # Delay between sub-regions
    
    print(f"Found {len(all_products)} total products for {name}")
    
    # Deduplicate by tile, preferring non-historical and newer dates
    # Extract tile name (e.g., n37w095) from URL and keep best version
    tile_pattern = re.compile(r'(n\d+w\d+|n\d+e\d+|s\d+w\d+|s\d+e\d+)', re.IGNORECASE)
    date_pattern = re.compile(r'_(\d{8})\.tif')
    
    tile_map = {}  # tile_name -> (product, is_historical, date)
    
    for p in all_products:
        url = p.get("downloadURL", "")
        
        # Extract tile name from URL
        match = tile_pattern.search(url)
        if not match:
            continue
        
        tile_name = match.group(1).lower()
        is_historical = "/historical/" in url
        
        # Extract date from filename
        date_match = date_pattern.search(url)
        file_date = date_match.group(1) if date_match else "00000000"
        
        if tile_name not in tile_map:
            tile_map[tile_name] = (p, is_historical, file_date)
        else:
            existing_product, existing_historical, existing_date = tile_map[tile_name]
            
            # Prefer non-historical over historical
            if existing_historical and not is_historical:
                tile_map[tile_name] = (p, is_historical, file_date)
            # If same historical status, prefer newer date
            elif existing_historical == is_historical and file_date > existing_date:
                tile_map[tile_name] = (p, is_historical, file_date)
    
    filtered = [item[0] for item in tile_map.values()]
    historical_used = sum(1 for item in tile_map.values() if item[1])
    
    print(f"After filtering: {len(filtered)} unique tiles")
    if historical_used > 0:
        print(f"  Using {historical_used} historical files (no current version available)")
    if len(all_products) - len(filtered) > 0:
        print(f"  Deduplicated {len(all_products) - len(filtered)} duplicate tiles")
    
    return filtered


def download_file(url: str, output_path: Path, retries: int = 3):
    """
    Download a file from URL with retry logic.
    
    Args:
        url: Download URL
        output_path: Local path to save file
        retries: Number of retry attempts
        
    Returns:
        True if successful, False if failed, None if 404 (not found)
    """
    for attempt in range(retries):
        try:
            print(f"Downloading: {output_path.name}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  Skipped (404 not found): {output_path.name}")
                return None  # Expected skip, not a failure
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
        except requests.RequestException as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    
    return False


def download_state_dem(
    name: str,
    output_dir: Path,
    max_workers: int = 4,
    bounds: tuple = None
) -> list:
    """
    Download all 1 arc-second DEM tiles for a state or region.
    
    Args:
        name: State code or region name
        output_dir: Directory to save downloaded files
        max_workers: Number of concurrent downloads
        bounds: Optional tuple of (minLon, minLat, maxLon, maxLat)
        
    Returns:
        List of downloaded file paths
    """
    state_dir = output_dir / name / "raw"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    products = get_dem_products(name, bounds=bounds)
    if not products:
        print(f"No products found for {name}")
        return []
    
    downloaded_files = []
    download_tasks = []
    
    for product in products:
        url = product.get("downloadURL")
        if not url:
            continue
            
        filename = url.split("/")[-1]
        output_path = state_dir / filename
        
        if output_path.exists():
            print(f"Already exists: {filename}")
            downloaded_files.append(output_path)
            continue
            
        download_tasks.append((url, output_path))
    
    if download_tasks:
        skipped_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_file, url, path): path
                for url, path in download_tasks
            }
            
            for future in as_completed(futures):
                path = futures[future]
                result = future.result()
                if result is True:
                    downloaded_files.append(path)
                elif result is None:
                    skipped_count += 1  # 404, already logged
                else:
                    failed_count += 1
                    print(f"Failed to download: {path.name}")
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files (404 not found)")
        if failed_count > 0:
            print(f"Failed to download {failed_count} files")
    
    return downloaded_files


def extract_tiffs(state_dir: Path) -> list:
    """
    Extract TIFF files from downloaded ZIP archives.
    
    Args:
        state_dir: Directory containing downloaded files
        
    Returns:
        List of extracted TIFF file paths
    """
    raw_dir = state_dir / "raw"
    extracted_dir = state_dir / "extracted"
    extracted_dir.mkdir(exist_ok=True)
    
    tiff_files = []
    
    for zip_file in raw_dir.glob("*.zip"):
        print(f"Extracting: {zip_file.name}")
        try:
            subprocess.run(
                ["unzip", "-o", "-j", str(zip_file), "*.tif", "-d", str(extracted_dir)],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            # Try extracting all files if specific pattern fails
            subprocess.run(
                ["unzip", "-o", str(zip_file), "-d", str(extracted_dir)],
                check=True,
                capture_output=True
            )
    
    # Find all TIFF files (including nested directories)
    for pattern in ["*.tif", "*.tiff", "**/*.tif", "**/*.tiff"]:
        tiff_files.extend(extracted_dir.glob(pattern))
    
    # Also check for already extracted TIFFs in raw directory
    for pattern in ["*.tif", "*.tiff"]:
        tiff_files.extend(raw_dir.glob(pattern))
    
    # Remove duplicates
    tiff_files = list(set(tiff_files))
    print(f"Found {len(tiff_files)} TIFF files")
    
    return tiff_files


def build_vrt(tiff_files: list, output_path: Path, nodata: float = -9999) -> bool:
    """
    Build a Virtual Raster (VRT) from multiple TIFF files.
    
    Args:
        tiff_files: List of input TIFF file paths
        output_path: Output VRT path
        nodata: NoData value
        
    Returns:
        True if successful
    """
    if not tiff_files:
        print("No TIFF files to combine")
        return False
    
    print(f"Building VRT from {len(tiff_files)} TIFF files...")
    
    # Create a text file listing all input TIFFs
    file_list = output_path.parent / "input_files.txt"
    with open(file_list, 'w') as f:
        for tiff in tiff_files:
            f.write(f"{tiff}\n")
    
    cmd = [
        "gdalbuildvrt",
        "-input_file_list", str(file_list),
        "-srcnodata", str(nodata),
        "-vrtnodata", str(nodata),
        str(output_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error building VRT: {result.stderr}")
        return False
    
    print(f"VRT created: {output_path}")
    return True


def create_scaled_vrt(
    input_path: Path,
    output_path: Path
) -> bool:
    """
    Create a VRT that applies 8-bit avarex encoding without creating a large TIFF.
    
    Encoding formula (from avarex elevation_tile_provider.dart):
        elevation_ft = (pixel_value * 80.4711845056) - 364.431597044586
        
    So to encode:
        pixel_value = (elevation_ft + 364.431597044586) / 80.4711845056
        
    This gives a range of approximately:
        pixel 0   = -364 ft (-111 m)
        pixel 255 = 20,156 ft (6,143 m)
    
    Args:
        input_path: Input DEM VRT/GeoTIFF path (elevation in meters)
        output_path: Output VRT path with 8-bit scaling
        
    Returns:
        True if successful
    """
    # Avarex encoding constants
    SLOPE = 80.4711845056
    INTERCEPT = -364.431597044586
    METERS_TO_FEET = 3.28084
    
    print("Creating scaled VRT with avarex encoding (no large TIFF)...")
    print(f"  Formula: pixel = (elevation_m * {METERS_TO_FEET} + {-INTERCEPT:.2f}) / {SLOPE:.2f}")
    print(f"  Range: pixel 0 = {INTERCEPT:.0f} ft, pixel 255 = {255 * SLOPE + INTERCEPT:.0f} ft")
    
    # Calculate scale parameters for gdal_translate
    # We need to find the meter values that map to 0 and 255
    # pixel 0: elevation_m = (0 * SLOPE + INTERCEPT) / METERS_TO_FEET
    # pixel 255: elevation_m = (255 * SLOPE + INTERCEPT) / METERS_TO_FEET
    src_min = (0 * SLOPE + INTERCEPT) / METERS_TO_FEET  # -111.07 m
    src_max = (255 * SLOPE + INTERCEPT) / METERS_TO_FEET  # 6143.35 m
    
    cmd = [
        "gdal_translate",
        "-of", "VRT",
        "-ot", "Byte",
        "-scale", str(src_min), str(src_max), "0", "255",
        str(input_path),
        str(output_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(f"stdout: {result.stdout}")
    if result.stderr:
        print(f"stderr: {result.stderr}")
    
    if result.returncode != 0:
        print(f"Error creating scaled VRT (exit code {result.returncode})")
        return False
    
    # Verify output file was created
    if not output_path.exists():
        print(f"Error: Output VRT was not created: {output_path}")
        return False
    
    print(f"Created scaled VRT: {output_path}")
    return True


def create_tiles(
    input_path: Path,
    output_dir: Path,
    tile_size: int = 512,
    zoom_levels: Optional[str] = None,
    processes: int = 4
) -> bool:
    """
    Create PNG web map tiles from a GeoTIFF using gdal2tiles.py.
    
    Args:
        input_path: Input GeoTIFF path
        output_dir: Directory to store tiles
        tile_size: Tile dimension in pixels (default 512)
        zoom_levels: Zoom levels to generate (e.g., "5-12"), auto if None
        processes: Number of parallel processes
        
    Returns:
        True if successful
    """
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {tile_size}x{tile_size} PNG tiles with gdal2tiles (no alpha)...")
    
    cmd = [
        "gdal2tiles.py",
        "--processes", str(processes),
        "--tilesize", str(tile_size),
        "-w", "all",
        "-r", "bilinear",
        "--tiledriver", "PNG",
        "-a", "0",
    ]
    
    if zoom_levels:
        cmd.extend(["-z", zoom_levels])
    
    cmd.extend([str(input_path), str(output_dir)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(f"stdout: {result.stdout}")
    if result.stderr:
        print(f"stderr: {result.stderr}")
    
    if result.returncode != 0:
        print(f"Error creating tiles (exit code {result.returncode})")
        return False
    
    tile_count = len(list(output_dir.rglob("*.png")))
    print(f"Created {tile_count} PNG tiles in {output_dir}")
    
    if tile_count == 0:
        print("Warning: No tiles were generated!")
        all_files = list(output_dir.rglob("*"))
        print(f"Files in output directory: {len(all_files)}")
        for f in all_files[:20]:
            print(f"  {f}")
        return False
    
    return True


def create_tiles_xyz(
    input_path: Path,
    output_dir: Path,
    tile_size: int = 512,
    zoom_levels: Optional[str] = None
) -> bool:
    """
    Create XYZ PNG tiles using gdal2tiles.
    
    Args:
        input_path: Input GeoTIFF path
        output_dir: Directory to store tiles
        tile_size: Tile dimension in pixels
        zoom_levels: Zoom levels to generate
        
    Returns:
        True if successful
    """
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating XYZ {tile_size}x{tile_size} PNG tiles (no alpha)...")
    
    cmd = [
        "gdal2tiles.py",
        "-p", "mercator",
        "--xyz",
        "--tilesize", str(tile_size),
        "-w", "all",
        "-r", "bilinear",
        "--tiledriver", "PNG",
        "-a", "0",
    ]
    
    if zoom_levels:
        cmd.extend(["-z", zoom_levels])
    
    cmd.extend([str(input_path), str(output_dir)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(f"stdout: {result.stdout}")
    if result.stderr:
        print(f"stderr: {result.stderr}")
    
    if result.returncode != 0:
        print(f"Error creating XYZ tiles (exit code {result.returncode})")
        return False
    
    tile_count = len(list(output_dir.rglob("*.png")))
    print(f"Created {tile_count} XYZ PNG tiles in {output_dir}")
    
    if tile_count == 0:
        print("Warning: No tiles were generated!")
        all_files = list(output_dir.rglob("*"))
        print(f"Files in output directory: {len(all_files)}")
        for f in all_files[:20]:
            print(f"  {f}")
        return False
    
    return True


def generate_manifest(tiles_dir: Path, manifest_path: Path) -> bool:
    """
    Generate a manifest file listing all tiles.
    
    Format:
        Line 1: 2603
        Line 2+: tiles/z/x/y.png
    
    Args:
        tiles_dir: Directory containing tiles
        manifest_path: Output manifest file path
        
    Returns:
        True if successful
    """
    print(f"Generating manifest: {manifest_path.name}")
    
    # Find all PNG tiles
    tile_files = []
    for png_file in tiles_dir.rglob("*.png"):
        # Get relative path from tiles_dir parent (to include "tiles/" prefix)
        rel_path = png_file.relative_to(tiles_dir.parent)
        tile_files.append(str(rel_path))
    
    # Sort files for consistent output
    tile_files.sort()
    
    print(f"Found {len(tile_files)} tile files")
    
    # Write manifest
    with open(manifest_path, 'w') as f:
        f.write("2603\n")
        for tile_path in tile_files:
            f.write(f"{tile_path}\n")
    
    print(f"Manifest written: {manifest_path}")
    return True


def create_zip(
    state: str,
    tiles_base: Path,
    manifest_path: Path,
    zip_path: Path
) -> bool:
    """
    Create a zip file containing tiles and manifest.
    
    Args:
        state: State code
        tiles_base: Base tiles directory (contains tiles/6/...)
        manifest_path: Path to manifest file
        zip_path: Output zip file path
        
    Returns:
        True if successful
    """
    print(f"Creating zip file: {zip_path.name}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            # Add manifest file
            zf.write(manifest_path, manifest_path.name)
            
            # Add all PNG tiles with relative paths from manifest location
            tile_count = 0
            for png_file in tiles_base.rglob("*.png"):
                rel_path = png_file.relative_to(tiles_base.parent)
                zf.write(png_file, str(rel_path))
                tile_count += 1
                if tile_count % 1000 == 0:
                    print(f"  Added {tile_count} tiles...")
            
            # Add openlayers.html if it exists
            openlayers_files = list(tiles_base.rglob("openlayers.html"))
            for html_file in openlayers_files:
                rel_path = html_file.relative_to(tiles_base.parent)
                zf.write(html_file, str(rel_path))
                print(f"  Added: {rel_path}")
            if not openlayers_files:
                print(f"  Warning: openlayers.html not found in {tiles_base}")
        
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Created zip: {zip_path} ({zip_size_mb:.1f} MB, {tile_count} tiles)")
        
        # Verify zip contents
        with zipfile.ZipFile(zip_path, 'r') as verify_zf:
            names = verify_zf.namelist()
            has_openlayers = any('openlayers.html' in n for n in names)
            print(f"  Zip contains {len(names)} files, openlayers.html: {has_openlayers}")
        
        return True
        
    except Exception as e:
        print(f"Error creating zip: {e}")
        return False


def process_state(
    name: str,
    output_dir: Path,
    tile_size: int = 512,
    skip_download: bool = False,
    skip_vrt: bool = False,
    max_workers: int = 4,
    zoom_levels: Optional[str] = None,
    xyz_tiles: bool = False,
    bounds: tuple = None
) -> bool:
    """
    Process a single state or region: download, build VRT, convert to 8-bit, and tile.
    
    Args:
        name: State code or region name
        output_dir: Base output directory
        tile_size: Tile dimension in pixels
        skip_download: Skip download step if files exist
        skip_vrt: Skip VRT build step if VRT exists
        max_workers: Number of concurrent downloads
        zoom_levels: Zoom levels for gdal2tiles (e.g., "1-10")
        xyz_tiles: Use XYZ tile naming convention
        bounds: Optional tuple of (minLon, minLat, maxLon, maxLat) for regions
        
    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    state_dir = output_dir / name
    state_dir.mkdir(parents=True, exist_ok=True)
    
    vrt_path = state_dir / f"{name}.vrt"
    scaled_path = state_dir / f"{name}_8bit.vrt"
    tiles_base = state_dir / "tiles"
    tiles_dir = tiles_base / "6"
    manifest_path = state_dir / f"{name}_ELEVATION"
    zip_path = state_dir / f"{name}_ELEVATION.zip"
    
    # Step 1: Download
    if not skip_download:
        downloaded = download_state_dem(name, output_dir, max_workers, bounds=bounds)
        if not downloaded:
            print(f"No files downloaded for {name}")
            return False
    
    # Step 2: Extract TIFFs
    tiff_files = extract_tiffs(state_dir)
    if not tiff_files:
        print(f"No TIFF files found for {name}")
        return False
    
    # Step 3: Build VRT (no merge, just virtual mosaic)
    if not skip_vrt or not vrt_path.exists():
        if not build_vrt(tiff_files, vrt_path):
            print(f"Failed to build VRT for {name}")
            return False
    else:
        print(f"Using existing VRT: {vrt_path}")
    
    # Step 4: Create scaled VRT with 8-bit avarex encoding (no large TIFF)
    if not scaled_path.exists() or not skip_vrt:
        if not create_scaled_vrt(vrt_path, scaled_path):
            print(f"Failed to create scaled VRT for {name}")
            return False
    else:
        print(f"Using existing scaled VRT: {scaled_path}")
    
    # Step 5: Create tiles using gdal2tiles (from 8-bit image)
    if xyz_tiles:
        tile_func = create_tiles_xyz
    else:
        tile_func = create_tiles
    
    if not tile_func(
        scaled_path,
        tiles_dir,
        tile_size=tile_size,
        zoom_levels=zoom_levels
    ):
        print(f"Failed to create tiles for {name}")
        return False
    
    # Step 6: Generate manifest file (use tiles_base for correct path format: tiles/6/z/x/y.png)
    if not generate_manifest(tiles_base, manifest_path):
        print(f"Failed to generate manifest for {name}")
        return False
    
    # Step 7: Create zip file
    if not create_zip(name, tiles_base, manifest_path, zip_path):
        print(f"Failed to create zip for {name}")
        return False
    
    print(f"\nCompleted processing for {name}")
    return True


def check_gdal_installation():
    """Check if GDAL tools are installed."""
    tools = ["gdalbuildvrt", "gdal_translate", "gdalinfo", "gdal2tiles.py", "gdal_calc.py"]
    missing = []
    
    for tool in tools:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            missing.append(tool)
    
    if missing:
        print(f"Error: Missing GDAL tools: {', '.join(missing)}")
        print("Install GDAL with: brew install gdal (macOS) or apt install gdal-bin python3-gdal (Linux)")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download USGS 1 arc-second DEM data and create tiles"
    )
    parser.add_argument(
        "-s", "--state",
        type=str,
        help="Single state code to process (e.g., CA, TX, NY)"
    )
    parser.add_argument(
        "states",
        nargs="*",
        default=[],
        help="State codes to process (e.g., CA TX NY) or ALL for all states"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./dem_data"),
        help="Output directory (default: ./dem_data)"
    )
    parser.add_argument(
        "-t", "--tile-size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step, use existing files"
    )
    parser.add_argument(
        "--skip-vrt",
        action="store_true",
        help="Skip VRT build step, use existing VRT and 8-bit files"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of concurrent downloads (default: 4)"
    )
    parser.add_argument(
        "-z", "--zoom",
        type=str,
        default="1-10",
        help="Zoom levels for gdal2tiles (default: '1-10')"
    )
    parser.add_argument(
        "--xyz",
        action="store_true",
        help="Use XYZ tile naming convention (default: TMS)"
    )
    parser.add_argument(
        "--list-states",
        action="store_true",
        help="List available state codes and exit"
    )
    parser.add_argument(
        "-r", "--region",
        type=str,
        help="Single region code to process (e.g., NW, SW, NE)"
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=[],
        help="Region codes to process (e.g., NW SW NE) or ALL for all regions"
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List available region codes and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_states:
        print("Available state codes:")
        for state in sorted(STATE_BOUNDS.keys()):
            bounds = STATE_BOUNDS[state]
            print(f"  {state}: ({bounds[0]:.2f}, {bounds[1]:.2f}) to ({bounds[2]:.2f}, {bounds[3]:.2f})")
        return 0
    
    if args.list_regions:
        print("Available region codes:")
        for region in sorted(REGION_BOUNDS.keys()):
            bounds = REGION_BOUNDS[region]
            print(f"  {region}: ({bounds[0]:.2f}, {bounds[1]:.2f}) to ({bounds[2]:.2f}, {bounds[3]:.2f})")
        return 0
    
    # Check GDAL installation
    if not check_gdal_installation():
        return 1
    
    # Determine what to process: regions or states
    process_regions = False
    targets = []
    
    # Check for region arguments first
    if args.region:
        region_code = args.region.upper()
        if region_code not in REGION_BOUNDS:
            print(f"Invalid region code: {region_code}")
            print("Use --list-regions to see available codes")
            return 1
        targets = [region_code]
        process_regions = True
    elif args.regions:
        if "ALL" in [r.upper() for r in args.regions]:
            targets = list(REGION_BOUNDS.keys())
        else:
            targets = [r.upper() for r in args.regions]
            invalid = [r for r in targets if r not in REGION_BOUNDS]
            if invalid:
                print(f"Invalid region codes: {', '.join(invalid)}")
                print("Use --list-regions to see available codes")
                return 1
        process_regions = True
    # Fall back to state arguments
    elif args.state:
        state_code = args.state.upper()
        if state_code not in STATE_BOUNDS:
            print(f"Invalid state code: {state_code}")
            print("Use --list-states to see available codes")
            return 1
        targets = [state_code]
    elif args.states:
        if "ALL" in [s.upper() for s in args.states]:
            targets = list(STATE_BOUNDS.keys())
        else:
            targets = [s.upper() for s in args.states]
            invalid = [s for s in targets if s not in STATE_BOUNDS]
            if invalid:
                print(f"Invalid state codes: {', '.join(invalid)}")
                print("Use --list-states to see available codes")
                return 1
    else:
        print("No state or region specified.")
        print("Use -s STATE or provide state codes as arguments.")
        print("Use -r REGION or --regions for regions.")
        print("Use --list-states or --list-regions to see available codes.")
        return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    target_type = "regions" if process_regions else "states"
    print(f"Processing {len(targets)} {target_type}")
    print(f"Output directory: {args.output}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Tile format: PNG (8-bit grayscale)")
    print(f"Zoom levels: {args.zoom}")
    
    # Process each target
    results = {}
    for target in targets:
        try:
            if process_regions:
                bounds = REGION_BOUNDS[target]
            else:
                bounds = None
            
            success = process_state(
                target,
                args.output,
                args.tile_size,
                args.skip_download,
                args.skip_vrt,
                args.workers,
                zoom_levels=args.zoom,
                xyz_tiles=args.xyz,
                bounds=bounds
            )
            results[target] = success
        except Exception as e:
            print(f"Error processing {target}: {e}")
            results[target] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [s for s, r in results.items() if r]
    failed = [s for s, r in results.items() if not r]
    
    print(f"Successful: {len(successful)} {target_type}")
    if successful:
        print(f"  {', '.join(successful)}")
    
    print(f"Failed: {len(failed)} {target_type}")
    if failed:
        print(f"  {', '.join(failed)}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
