# OPCardOCR

OPCardOCR extracts One Piece TCG card codes (e.g., `OP01-001`, `EB02-045`) from photos of binder pages. It detects the page boundary, corrects perspective, slices the image into a 3×3 grid, identifies occupied pockets, and runs OCR on the bottom-right code area of each card to produce a JSON report of detected codes.

## How It Works

1. **Page detection + perspective warp**: Finds the outer binder page contour and warps it into a flat rectangle.
2. **Grid slicing**: Divides the warped page into a fixed 3×3 grid of slots.
3. **Occupancy check**: Uses edge density to decide whether a slot contains a card.
4. **Code ROI extraction**: Crops the bottom-right region of each occupied card.
5. **OCR + normalization**: Runs Tesseract OCR and normalizes results to a canonical code format.
6. **JSON output**: Writes per-image results and a unique list of all codes found.

## Dependencies

### Python packages

Install via pip:

```bash
pip install opencv-python numpy pytesseract
```

### System dependency (Tesseract)

You must install the Tesseract binary on your system:

- **macOS (Homebrew)**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download the installer from the Tesseract project releases and ensure `tesseract` is in your PATH.

## Usage

### Basic run

```bash
python OPCardOCR.py --input /path/to/binder_pages --output cards.json
```

### Optional arguments

- `--rows` / `--cols`: grid size (default `3x3`)
- `--min_conf`: minimum confidence to keep codes (default `0.70`)
- `--debug`: save warped images and OCR crops under `<input>/_debug/<image_name>/`

Example:

```bash
python OPCardOCR.py \
  --input ./testIMG \
  --output cards.json \
  --min_conf 0.75 \
  --debug
```

## Output Format

The script writes a JSON file with:

- `source_folder`: absolute path to the input folder
- `images`: per-image results, including slot metadata and OCR results
- `all_codes`: all kept codes in scan order
- `unique_codes`: sorted unique codes

Example (abbreviated):

```json
{
  "source_folder": "/abs/path/to/binder_pages",
  "images": [
    {
      "image": "page1.jpg",
      "grid": { "rows": 3, "cols": 3 },
      "cards": [
        {
          "index": 1,
          "row": 1,
          "col": 1,
          "occupied": true,
          "code": "OP01-001",
          "confidence": 0.85
        }
      ]
    }
  ],
  "all_codes": ["OP01-001"],
  "unique_codes": ["OP01-001"]
}
```

## Tips for Best Results

- Use **well-lit, sharp photos** with minimal glare.
- Ensure the entire binder page is visible in the image.
- If OCR results look off, run with `--debug` to inspect the crops and tune lighting.

## Notes

- The OCR normalization only accepts `OP` and `EB` prefixes by default.
- If you need to limit allowed set IDs, edit `VALID_SET_IDS` in `OPCardOCR.py`.

## License

See `LICENSE.txt`.
