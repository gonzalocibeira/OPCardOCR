#!/usr/bin/env python3
"""
One Piece TCG binder page -> card code extractor
- Input: folder of page photos (each page is a 3x3 grid; some slots empty)
- Output: JSON "database" of detected card codes per image + unique codes

Pipeline:
1) Detect page boundary + perspective warp to a flat rectangle
2) Slice into 3x3 cells (fixed grid)
3) For each cell, decide if it's occupied
4) If occupied, crop bottom-right "code ROI"
5) OCR with Tesseract + strict regex normalization
6) Save JSON (and optional debug crops)

Dependencies:
  pip install opencv-python numpy pytesseract
System:
  Install Tesseract binary (brew install tesseract / apt install tesseract-ocr / Windows installer)
"""

import os
import re
import json
import argparse
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
import pytesseract


# Accept OPxx-xxx / EBxx-xxx (robust to OCR spacing/dashes)
CODE_RE = re.compile(r"(OP|EB)\s*0?(\d{1,2})\s*[-–—]?\s*(\d{3})", re.IGNORECASE)

# If you only want to allow certain sets, fill this. Otherwise leave None.
VALID_PREFIXES = {"OP", "EB"}
VALID_SET_IDS = None  # e.g. {"OP01","OP02","OP03","OP04","OP05","OP06","OP07","OP08","OP09","OP10","OP11","OP12","OP13","OP14","EB01","EB02","EB03"}


def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # tr
    rect[3] = pts[np.argmax(d)]  # bl
    return rect


def four_point_warp(image: np.ndarray, pts: np.ndarray, out_w: int = 1800) -> np.ndarray:
    rect = order_points(pts.astype(np.float32))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Normalize output size: fixed width, keep aspect
    if maxWidth <= 0 or maxHeight <= 0:
        return image
    scale = out_w / float(maxWidth)
    out_h = int(maxHeight * scale)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


def detect_page_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """Find the outer page contour as a 4-point polygon."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15 * (img.shape[0] * img.shape[1]):
            return approx.reshape(4, 2)

    # Fallback: use minAreaRect on biggest contour
    c = contours[0]
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def slice_grid(warped: np.ndarray, rows: int = 3, cols: int = 3, margin: float = 0.01) -> List[Tuple[int,int,int,int]]:
    """Return list of cell rectangles (x,y,w,h) in reading order."""
    h, w = warped.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    usable_w = w - 2 * mx
    usable_h = h - 2 * my

    cell_w = usable_w / cols
    cell_h = usable_h / rows

    boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(mx + c * cell_w)
            y1 = int(my + r * cell_h)
            x2 = int(mx + (c + 1) * cell_w)
            y2 = int(my + (r + 1) * cell_h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def is_occupied(cell_bgr: np.ndarray) -> bool:
    """
    Heuristic to decide if slot contains a card:
    - Empty pocket is mostly uniform/dark texture -> low edge density
    - Card has artwork/text -> higher edge density
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    return edge_ratio > 0.025  # tune if needed


def normalize_code(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.upper().strip()
    # common OCR confusions
    t = t.replace("—", "-").replace("–", "-")
    t = t.replace("0P", "OP")
    t = t.replace("E8", "EB")
    # sometimes OCR inserts spaces
    t = re.sub(r"\s+", " ", t)

    m = CODE_RE.search(t)
    if not m:
        return None

    prefix = m.group(1).upper()
    if prefix not in VALID_PREFIXES:
        return None

    set_num = int(m.group(2))
    card_num = m.group(3)

    code = f"{prefix}{set_num:02d}-{card_num}"

    if VALID_SET_IDS is not None:
        set_id = code.split("-")[0]
        if set_id not in VALID_SET_IDS:
            return None

    return code


def extract_code_from_cell(cell_bgr: np.ndarray, debug_dir: Optional[str], tag: str) -> Tuple[Optional[str], float]:
    """
    Crop bottom-right region, preprocess, OCR, normalize.
    Returns (code, confidence-ish score).
    """
    h, w = cell_bgr.shape[:2]

    def detect_card_bounds() -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 140)
        edges = cv2.dilate(edges, None, iterations=2)
        edges = cv2.erode(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cell_area = float(h * w)
        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.35 * cell_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if cw == 0 or ch == 0:
                continue
            aspect = cw / float(ch)
            if not (0.55 <= aspect <= 0.95):
                continue
            if area > best_area:
                best_area = area
                best = (x, y, cw, ch)

        return best

    card_bounds = detect_card_bounds()
    if card_bounds:
        cx, cy, cw, ch = card_bounds
    else:
        cx, cy, cw, ch = 0, 0, w, h

    # Crop more forgiving bottom-right ROI from detected card bounds
    x1 = cx + int(cw * 0.60)
    y1 = cy + int(ch * 0.78)
    x2 = cx + cw
    y2 = cy + ch
    pad = max(2, int(min(cw, ch) * 0.02))
    x1 = max(cx, x1 - pad)
    y1 = max(cy, y1 - pad)
    x2 = min(cx + cw, x2 + pad)
    y2 = min(cy + ch, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = int(w * 0.60), int(h * 0.78), w, h
    roi = cell_bgr[y1:y2, x1:x2]

    # Upscale for OCR
    roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    def build_variants(channel: np.ndarray, label: str) -> List[Tuple[str, np.ndarray]]:
        variants: List[Tuple[str, np.ndarray]] = []
        blur = cv2.GaussianBlur(channel, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)

        variants.append((f"{label}_otsu", cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
        variants.append((f"{label}_clahe_otsu", cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
        variants.append((f"{label}_adapt", cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
        )))
        variants.append((f"{label}_clahe_adapt", cv2.adaptiveThreshold(
            clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
        )))
        return variants

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    channels = [
        ("gray", gray),
        ("hsv_v", hsv[:, :, 2]),
        ("lab_l", lab[:, :, 0]),
    ]

    candidates: List[Tuple[str, np.ndarray]] = []
    for label, channel in channels:
        candidates.extend(build_variants(channel, label))

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=OPBE0123456789-"
    best_code: Optional[str] = None
    best_text = ""
    best_score = -1.0
    best_thr: Optional[np.ndarray] = None
    best_label = ""

    for label, thr in candidates:
        cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        text = pytesseract.image_to_string(cleaned, config=config).strip()
        code = normalize_code(text)
        score = 0.0
        if code:
            score = 1.0
            if re.search(r"[^A-Z0-9\-\s]", text.upper()):
                score -= 0.15
        if score > best_score:
            best_score = score
            best_code = code
            best_text = text
            best_thr = cleaned
            best_label = label

    conf = 0.0
    if best_code:
        conf = 0.85
        if re.search(r"[^A-Z0-9\-\s]", best_text.upper()):
            conf = 0.70

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_roi.png"), roi)
        if best_thr is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{tag}_thr_{best_label}.png"), best_thr)

    return best_code, conf


def process_image(path: str, rows: int, cols: int, debug: bool) -> Dict[str, Any]:
    img = cv2.imread(path)
    if img is None:
        return {"image": os.path.basename(path), "error": "failed_to_read", "cards": []}

    quad = detect_page_quad(img)
    if quad is None:
        warped = img
    else:
        warped = four_point_warp(img, quad, out_w=1800)

    boxes = slice_grid(warped, rows=rows, cols=cols, margin=0.01)

    debug_dir = None
    if debug:
        debug_dir = os.path.join(os.path.dirname(path), "_debug", os.path.splitext(os.path.basename(path))[0])
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "warped.png"), warped)

    cards = []
    found_codes = set()

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cell = warped[y:y+h, x:x+w]
        occ = is_occupied(cell)

        code = None
        conf = 0.0

        if occ:
            tag = f"cell_{idx:02d}"
            code, conf = extract_code_from_cell(cell, debug_dir, tag)

            # Deduplicate within the same page
            if code and code in found_codes:
                code = None
                conf = 0.0
            elif code:
                found_codes.add(code)

        cards.append({
            "index": idx,
            "row": (idx - 1) // cols + 1,
            "col": (idx - 1) % cols + 1,
            "occupied": bool(occ),
            "code": code,
            "confidence": conf
        })

        if debug and idx <= 10:
            # save a few sample cell crops for sanity
            cv2.imwrite(os.path.join(debug_dir, f"sample_cell_{idx:02d}.png"), cell)

    return {
        "image": os.path.basename(path),
        "path": os.path.abspath(path),
        "grid": {"rows": rows, "cols": cols},
        "cards": cards
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with binder page photos")
    ap.add_argument("--output", default="cards.json", help="Output JSON file")
    ap.add_argument("--rows", type=int, default=3, help="Grid rows (default 3)")
    ap.add_argument("--cols", type=int, default=3, help="Grid cols (default 3)")
    ap.add_argument("--min_conf", type=float, default=0.70, help="Keep codes with confidence >= this")
    ap.add_argument("--debug", action="store_true", help="Write debug warped/crops next to images in _debug/")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    images = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if f.lower().endswith(exts)
    ]

    db: Dict[str, Any] = {
        "source_folder": os.path.abspath(args.input),
        "images": [],
        "all_codes": [],
        "unique_codes": []
    }

    all_codes: List[str] = []
    uniq = set()

    for p in images:
        res = process_image(p, rows=args.rows, cols=args.cols, debug=args.debug)

        for c in res["cards"]:
            if c["code"] and c["confidence"] >= args.min_conf:
                all_codes.append(c["code"])
                uniq.add(c["code"])
            else:
                # if low confidence, blank it so it shows up for manual review
                if c["code"] and c["confidence"] < args.min_conf:
                    c["code"] = None

        db["images"].append(res)

    db["all_codes"] = all_codes
    db["unique_codes"] = sorted(uniq)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output}")
    print(f"Codes kept: {len(all_codes)}  (unique: {len(uniq)})")
    if args.debug:
        print("Debug written under <input>/_debug/<image_name>/")

if __name__ == "__main__":
    main()
