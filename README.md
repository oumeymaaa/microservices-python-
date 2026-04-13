# Qari OCR Test

This is a separate test project for:

- base model: `Qwen/Qwen2-VL-2B-Instruct`
- OCR adapter: `NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct`

It is independent from the Tesseract project.

## What this script does

- loads one image
- prepares a smaller copy for faster inference
- loads the base model and the Qari adapter
- runs one strict OCR-style prompt
- saves the result to a JSON file

## Files

- `main.py`: main script
- `requirements.txt`: dependencies
- `prepared_image.jpg`: prepared image used for inference
- `result.json`: saved result

## Install

Use a Python environment, then run:

```powershell
pip install -r requirements.txt
```

If you want GPU acceleration, make sure PyTorch is installed with CUDA support.

## Usage

Default image and default output:

```powershell
python main.py
```

Specific image:

```powershell
python main.py ..\cin2.jpg
```

Specific image and output:

```powershell
python main.py ..\cin2.jpg my_result.json
```

Specific image, output, image size, and max tokens:

```powershell
python main.py ..\cin2.jpg my_result.json 768 48
```

## Command line arguments

The script accepts up to 4 arguments:

1. image path
2. output JSON path
3. max image size
4. max new tokens

## Notes

- GPU is much faster than CPU for this model.
- Tesseract is still much faster overall.
- This model can produce better Arabic OCR on some images, but it can also hallucinate.
