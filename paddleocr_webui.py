import os
import sys
import argparse
import time
import tempfile
import math
import gc
import io as io_module
import numpy as np
from pathlib import Path

os.environ["FLAGS_use_cuda_managed_memory"] = "false"
os.environ["PADDLE_DISABLE_GPU_MEMORY_GROWTH"] = "true"
os.environ["PADDLE_ALLOW_GROWTH"] = "true"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PADDLE_LAYER_CACHE_PATH"] = os.path.join(os.path.dirname(__file__), ".paddle_cache")

_PDX_CACHE = os.path.join(os.path.dirname(__file__), ".paddlex")
os.environ["PADDLE_PDX_CACHE_HOME"] = _PDX_CACHE
os.environ["HF_HOME"] = os.path.join(_PDX_CACHE, "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(_PDX_CACHE, "huggingface", "hub")
os.makedirs(_PDX_CACHE, exist_ok=True)

try:
    import paddle
    GPU_AVAILABLE = paddle.device.is_compiled_with_cuda()
    GPU_COUNT = paddle.device.cuda.device_count() if GPU_AVAILABLE else 0
    if GPU_AVAILABLE and GPU_COUNT > 0:
        paddle.device.set_device("gpu:0")
    GPU_NAME = paddle.device.cuda.get_device_name(0) if GPU_AVAILABLE and GPU_COUNT > 0 else "N/A"
except Exception:
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    GPU_NAME = "N/A"

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio pillow")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not installed. Run: pip install pillow")
    sys.exit(1)

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from paddlex import create_pipeline
    PADDLEX_AVAILABLE = True
except ImportError:
    PADDLEX_AVAILABLE = False

pipeline = None
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_device_str():
    if GPU_AVAILABLE and GPU_COUNT > 0:
        return "gpu:0"
    return "cpu"

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return "Pipeline already loaded"
    if not PADDLEX_AVAILABLE:
        return "Error: PaddleX not installed"
    device = get_device_str()
    try:
        pipeline = create_pipeline(pipeline="OCR", device=device)
        return f"Pipeline loaded on {device}"
    except Exception as e:
        return f"Error loading pipeline: {str(e)}"

def draw_boxes(image, dt_polys, texts, scores):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
    for poly, text, score in zip(dt_polys, texts, scores):
        points = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(points, outline=(0, 255, 0), width=2)
        label = f"{text} ({score:.2f})"
        x, y = int(poly[0][0]), max(0, int(poly[0][1]) - 20)
        draw.rectangle([x, y, x + len(label) * 10, y + 22], fill=(0, 0, 0))
        draw.text((x + 2, y + 1), label, fill=(255, 255, 255), font=font)
    return img

def extract_result_texts(result):
    res = result.get("res", result)
    dt_polys = res.get("dt_polys", [])
    rec_texts = res.get("rec_texts", [])
    rec_scores = res.get("rec_scores", [])
    if isinstance(dt_polys, np.ndarray):
        dt_polys = dt_polys.tolist()
    return dt_polys, rec_texts, rec_scores

def save_text_file(text, filename_prefix, ext=".txt"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{filename_prefix}_{ts}{ext}"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    return fpath

def save_image_file(image, filename_prefix):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{filename_prefix}_{ts}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    image.save(fpath, "PNG")
    return fpath

def poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def _build_line_items(dt_polys, rec_texts, rec_scores):
    items = []
    heights = []
    for i, (poly, text, score) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
        if not text or not str(text).strip():
            continue
        x1, y1, x2, y2 = poly_bbox(poly)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bh = y2 - y1
        heights.append(bh)
        items.append({
            "text": str(text).strip(),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy, "height": bh, "width": x2 - x1,
            "score": score, "index": i
        })
    return items, heights

def _group_into_lines(items, line_tolerance):
    if not items:
        return []
    items.sort(key=lambda it: (it["cy"], it["cx"]))
    lines = []
    current_line = [items[0]]
    for item in items[1:]:
        prev = current_line[-1]
        if abs(item["cy"] - prev["cy"]) < line_tolerance or (
           abs(item["y1"] - prev["y1"]) < line_tolerance and
           abs(item["y2"] - prev["y2"]) < line_tolerance):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)
    for line in lines:
        line.sort(key=lambda it: it["x1"])
    return lines

def _detect_table_regions(lines, avg_height):
    if len(lines) < 2:
        return {}
    table_rows = {}
    table_id = 0
    in_table = False
    table_col_counts = []

    for i, line in enumerate(lines):
        if len(line) < 2:
            if in_table:
                if len(table_col_counts) >= 2:
                    for ti in table_rows.get(table_id, []):
                        pass
                table_id += 1
            in_table = False
            table_col_counts = []
            continue

        inter_gaps = []
        for j in range(1, len(line)):
            gap = line[j]["x1"] - line[j-1]["x2"]
            inter_gaps.append(gap)

        has_table_gaps = any(g > avg_height * 2.5 for g in inter_gaps) if inter_gaps else False
        all_small = all(g < avg_height * 0.6 for g in inter_gaps) if inter_gaps else False

        if not has_table_gaps:
            if in_table:
                if len(table_col_counts) >= 2 and len(set(table_col_counts)) <= 2:
                    pass
                else:
                    for ti in list(table_rows.keys()):
                        if table_id in table_rows and len(table_rows[table_id]) < 2:
                            del table_rows[table_id]
                table_id += 1
            in_table = False
            table_col_counts = []
            continue

        if not in_table:
            in_table = True
            table_rows.setdefault(table_id, [])

        table_rows[table_id].append(i)
        table_col_counts.append(len(line))

    if in_table:
        if len(table_col_counts) < 2:
            if table_id in table_rows:
                del table_rows[table_id]

    return table_rows

def _format_table(lines_in_table, avg_height):
    if len(lines_in_table) < 2:
        return None

    all_line_items = []
    for line in lines_in_table:
        all_line_items.extend(line)

    all_x1 = sorted(set(round(it["x1"], 1) for it in all_line_items))

    col_boundaries = []
    current = [all_x1[0]]
    for v in all_x1[1:]:
        if v - current[-1] > avg_height * 2:
            col_boundaries.append((min(current), max(current) + avg_height * 0.5))
            current = [v]
        else:
            current.append(v)
    col_boundaries.append((min(current), max(current) + avg_height * 0.5))

    if len(col_boundaries) < 2:
        return None

    for cb in col_boundaries:
        if cb[1] <= cb[0]:
            return None

    num_cols = len(col_boundaries)
    aligned = True
    for line in lines_in_table:
        count = 0
        for cb in col_boundaries:
            for it in line:
                if cb[0] <= it["cx"] <= cb[1]:
                    count += 1
                    break
        if count == 0:
            aligned = False
            break

    if not aligned:
        return None

    rows = []
    header_row = lines_in_table[0]
    header_cells = [""] * num_cols
    for ci, cb in enumerate(col_boundaries):
        for it in header_row:
            if cb[0] <= it["cx"] <= cb[1]:
                header_cells[ci] = it["text"]
                break
    header_text = " | ".join(header_cells)

    has_real_header = (
        sum(it["height"] for it in header_row) / len(header_row) > avg_height * 1.1
        or len(header_row) == num_cols
    )

    md_table = []
    if has_real_header:
        md_table.append(f"| {header_text} |")
    else:
        rows.append(header_cells)
        start_row = 1
        md_table.append(f"| {header_text} |")

    md_table.append(f"|{'---|' * num_cols}")

    data_start = 1 if has_real_header else 0
    for line in lines_in_table[data_start:]:
        cells = [""] * num_cols
        for ci, cb in enumerate(col_boundaries):
            for it in line:
                if cb[0] <= it["cx"] <= cb[1]:
                    cells[ci] = it["text"]
                    break
        md_table.append(f"| {' | '.join(cells)} |")

    return "\n".join(md_table)

def _is_list_marker(text):
    text = text.strip()
    markers = [
        u"\u2022", u"\u25cf", u"\u25cb", u"\u25a0", u"\u25b6",
        "-", "*", "+",
    ]
    for m in markers:
        if text == m or text.startswith(m + " ") or text.startswith(m + "\t"):
            return True, m
    import re
    m = re.match(r'^(\d+[.)]\s)', text)
    if m:
        return True, m.group(1).strip()
    m = re.match(r'^([a-zA-Z][.)]\s)', text)
    if m:
        return True, m.group(1).strip()
    return False, None

def _classify_line(line, avg_height, page_width, prev_line_ys, prev_last_line, all_lines_info):
    line_avg_h = sum(it["height"] for it in line) / len(line) if line else avg_height
    line_text = " ".join(it["text"] for it in line)
    line_y = sum(it["cy"] for it in line) / len(line)

    is_centered = False
    if page_width and len(line) <= 5:
        line_cx = sum(it["cx"] for it in line) / len(line) if line else 0
        is_centered = abs(line_cx - page_width / 2) < page_width * 0.15

    is_header = False
    header_level = 2
    if line_avg_h > avg_height * 2.0 and len(line_text) < 50:
        is_header = True
        header_level = 1
    elif line_avg_h > avg_height * 1.6 and len(line_text) < 60:
        is_header = True
        header_level = 2
    elif line_avg_h > avg_height * 1.35 and len(line_text) < 70:
        is_header = True
        header_level = 3
    elif is_centered and len(line_text) < 50 and line_avg_h > avg_height * 1.1:
        is_header = True
        header_level = 2
    elif len(line) == 1 and len(line_text) < 35 and line_avg_h > avg_height * 1.15:
        is_header = True
        header_level = 3

    is_bold = line_avg_h > avg_height * 1.15 and not is_header

    is_list = False
    list_marker = None
    if len(line) >= 1:
        is_list, list_marker = _is_list_marker(line[0]["text"])

    is_new_paragraph = False
    if prev_line_ys and len(prev_line_ys) >= 2:
        recent_gaps = [prev_line_ys[j] - prev_line_ys[j-1] for j in range(1, len(prev_line_ys))]
        if recent_gaps:
            avg_gap = sum(recent_gaps) / len(recent_gaps)
            if prev_line_ys and (line_y - prev_line_ys[-1]) > avg_gap * 2.2:
                is_new_paragraph = True

    return {
        "text": line_text,
        "is_header": is_header,
        "header_level": header_level,
        "is_bold": is_bold,
        "is_centered": is_centered,
        "is_list": is_list,
        "list_marker": list_marker,
        "is_new_paragraph": is_new_paragraph,
        "line_y": line_y,
        "line_avg_h": line_avg_h,
    }

def reconstruct_markdown(dt_polys, rec_texts, rec_scores, page_width=None, page_height=None):
    if not dt_polys or not rec_texts:
        return ""

    items, heights = _build_line_items(dt_polys, rec_texts, rec_scores)
    if not items:
        return ""

    avg_height = sum(heights) / len(heights)
    lines = _group_into_lines(items, avg_height * 0.6)

    table_regions = _detect_table_regions(lines, avg_height)

    table_line_indices = set()
    for tid, line_indices in table_regions.items():
        table_line_indices.update(line_indices)

    all_line_ys = [sum(it["cy"] for it in line) / len(line) for line in lines]

    md_lines = []
    prev_line_ys = []
    table_lines_buffer = []
    current_table_id = None
    i = 0

    while i < len(lines):
        line = lines[i]
        line_y = all_line_ys[i]

        if i in table_line_indices:
            tid = None
            for t_id, indices in table_regions.items():
                if i in indices:
                    tid = t_id
                    break

            if current_table_id is None:
                table_lines_buffer = [line]
                current_table_id = tid
            elif tid == current_table_id:
                table_lines_buffer.append(line)
            else:
                tbl = _format_table(table_lines_buffer, avg_height)
                if tbl:
                    if md_lines and md_lines[-1] != "":
                        md_lines.append("")
                    md_lines.append(tbl)
                    md_lines.append("")
                table_lines_buffer = [line]
                current_table_id = tid

            prev_line_ys.append(line_y)
            i += 1
            continue
        else:
            if table_lines_buffer:
                tbl = _format_table(table_lines_buffer, avg_height)
                if tbl:
                    if md_lines and md_lines[-1] != "":
                        md_lines.append("")
                    md_lines.append(tbl)
                    md_lines.append("")
                table_lines_buffer = []
                current_table_id = None

        cls = _classify_line(line, avg_height, page_width, prev_line_ys, None, None)

        if cls["is_new_paragraph"] and md_lines and md_lines[-1] != "":
            md_lines.append("")

        if cls["is_header"]:
            prefix = "#" * min(cls["header_level"], 3)
            text = cls["text"]
            md_lines.append(f"{prefix} {text}")
        elif cls["is_list"]:
            prefix = "  " if prev_line_ys else ""
            text = cls["text"]
            marker = cls["list_marker"]
            if marker and text.startswith(marker):
                rest = text[len(marker):].strip()
                md_lines.append(f"{prefix}- {rest if rest else text}")
            else:
                md_lines.append(f"{prefix}- {text}")
        else:
            text = cls["text"]
            if cls["is_bold"]:
                text = f"**{text}**"
            md_lines.append(text)

        prev_line_ys.append(line_y)
        i += 1

    if table_lines_buffer:
        tbl = _format_table(table_lines_buffer, avg_height)
        if tbl:
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            md_lines.append(tbl)

    return "\n".join(md_lines)

def ocr_recognize(image, show_boxes):
    if image is None:
        return None, "Please upload an image", "", None, None, None, None

    load_result = load_pipeline()
    if "Error" in load_result:
        return None, load_result, "", None, None, None, None

    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_w, img_h = image.size

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path)

        start_time = time.time()
        results = list(pipeline.predict(input=tmp_path))
        elapsed = time.time() - start_time

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        if not results:
            return image, "No text detected", "", None, None, None, None

        result = results[0]
        dt_polys, rec_texts, rec_scores = extract_result_texts(result)

        if not rec_texts:
            return image, "No text detected", "", None, None, None, None

        result_image = image
        annotated = None
        if show_boxes and dt_polys and rec_texts and rec_scores:
            result_image = draw_boxes(image, dt_polys, rec_texts, rec_scores)
            annotated = result_image

        full_text = "\n".join(rec_texts)
        md_text = reconstruct_markdown(dt_polys, rec_texts, rec_scores, img_w, img_h)

        num_chars = sum(len(str(t)) for t in rec_texts)
        device_name = GPU_NAME if (GPU_AVAILABLE and GPU_COUNT > 0) else "CPU"
        stats = f"Device: {device_name} | Time: {elapsed:.2f}s | Regions: {len(rec_texts)} | Chars: {num_chars}"

        txt_path = save_text_file(full_text, "ocr_image")
        md_path = save_text_file(md_text, "ocr_image", ".md")
        img_path = save_image_file(result_image, "ocr_image") if annotated else None

        return result_image, stats, full_text, txt_path, img_path, md_path

    except Exception as e:
        import traceback
        return image, f"Error: {str(e)}", traceback.format_exc(), None, None, None, None

def download_from_state(text, ext=".txt"):
    if not text:
        return None
    return save_text_file(text, "ocr_export", ext)

def download_image(img):
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return save_image_file(img, "ocr_export")

def ocr_pdf(pdf_file, show_boxes, progress=gr.Progress()):
    if pdf_file is None:
        return None, "Please upload a PDF file", "", None, None, None

    if not PYMUPDF_AVAILABLE:
        return None, "PyMuPDF not installed", "", None, None, None

    load_result = load_pipeline()
    if "Error" in load_result:
        return None, load_result, "", None, None, None

    try:
        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        progress(0, desc=f"Opening PDF: {total_pages} pages, {file_size_mb:.1f}MB")

        dpi = 200
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        all_texts = []
        all_md_parts = []
        all_images = []

        for page_num in range(total_pages):
            progress((page_num + 1) / total_pages, desc=f"Processing page {page_num + 1}/{total_pages}...")

            page = doc[page_num]
            page_rect = page.rect
            pw, ph = page_rect.width * zoom, page_rect.height * zoom

            fitz_text = page.get_text("text").strip()
            has_embedded_text = len(fitz_text) > 50

            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            page_image = Image.open(io_module.BytesIO(img_data))

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                page_image.save(tmp_path, "PNG")

            results = list(pipeline.predict(input=tmp_path))
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            if not results or not extract_result_texts(results[0])[1]:
                page_label = f"--- Page {page_num + 1} ---"
                if has_embedded_text:
                    all_texts.append(f"{page_label}\n{fitz_text}")
                    all_md_parts.append(fitz_text)
                else:
                    all_texts.append(f"{page_label}\n(No text detected)")
                    all_md_parts.append(f"(No text detected)")
                if len(all_images) < 5:
                    all_images.append(page_image)
                page = None
                pix = None
                page_image = None
                continue

            result = results[0]
            dt_polys, rec_texts, rec_scores = extract_result_texts(result)

            if not rec_texts:
                page_label = f"--- Page {page_num + 1} ---"
                if has_embedded_text:
                    all_texts.append(f"{page_label}\n{fitz_text}")
                    all_md_parts.append(fitz_text)
                else:
                    all_texts.append(f"{page_label}\n(No text detected)")
                    all_md_parts.append(f"(No text detected)")
                if len(all_images) < 5:
                    all_images.append(page_image)
            else:
                page_label = f"--- Page {page_num + 1} ---"
                all_texts.append(f"{page_label}\n" + "\n".join(rec_texts))

                ocr_md = reconstruct_markdown(dt_polys, rec_texts, rec_scores, pw, ph)
                if has_embedded_text and len(fitz_text) > len("\n".join(rec_texts)) * 0.5:
                    all_md_parts.append(fitz_text)
                else:
                    all_md_parts.append(ocr_md)

                if show_boxes and dt_polys and rec_scores and len(all_images) < 5:
                    annotated = draw_boxes(page_image, dt_polys, rec_texts, rec_scores)
                    all_images.append(annotated)
                elif len(all_images) < 5:
                    all_images.append(page_image)

            page = None
            pix = None
            page_image = None
            gc.collect()

        doc.close()

        full_text = "\n\n".join(all_texts)
        full_md = "\n\n".join(all_md_parts)
        num_chars = sum(len(str(t)) for t in full_text.split("\n"))

        device_name = GPU_NAME if (GPU_AVAILABLE and GPU_COUNT > 0) else "CPU"
        stats = f"Device: {device_name} | Pages: {total_pages} | Size: {file_size_mb:.1f}MB | Chars: {num_chars}"

        preview_img = all_images[0] if all_images else None
        preview_text = full_text[:4000]
        if len(full_text) > 4000:
            preview_text += f"\n\n... (preview truncated, {len(full_text)} chars total)"

        txt_path = save_text_file(full_text, "ocr_pdf")
        md_path = save_text_file(full_md, "ocr_pdf", ".md")

        return preview_img, stats, preview_text, txt_path, md_path

    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}", traceback.format_exc(), None, None, None

def batch_ocr(files, progress=gr.Progress()):
    if not files:
        return "Please upload files", None, None

    load_result = load_pipeline()
    if "Error" in load_result:
        return load_result, None, None

    all_results = []
    all_md_parts = []
    total = len(files)

    for i, file_obj in enumerate(files):
        progress((i + 1) / total, desc=f"Processing {i + 1}/{total}...")
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        ext = os.path.splitext(file_path)[1].lower()
        basename = os.path.basename(file_path)

        if ext == ".pdf":
            if not PYMUPDF_AVAILABLE:
                hdr = f"\n{'='*60}\nFile: {basename}\n{'='*60}"
                all_results.append(f"{hdr}\nPDF support requires: pip install pymupdf")
                all_md_parts.append(f"# {basename}\n\n*PDF support not available*")
                continue

            try:
                doc = fitz.open(file_path)
                hdr = f"\n{'='*60}\nFile: {basename} ({doc.page_count} pages)\n{'='*60}"
                all_results.append(hdr)
                all_md_parts.append(f"# {basename}\n")

                for pn in range(doc.page_count):
                    page = doc[pn]
                    fitz_text = page.get_text("text").strip()
                    has_text = len(fitz_text) > 50

                    pix = page.get_pixmap(dpi=150)
                    img_data = pix.tobytes("png")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        with open(tmp_path, "wb") as f:
                            f.write(img_data)
                    results = list(pipeline.predict(input=tmp_path))
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                    if results:
                        dt_polys, rec_texts, rec_scores = extract_result_texts(results[0])
                        if rec_texts:
                            all_results.append(f"\n--- Page {pn + 1} ---")
                            all_results.append("\n".join(rec_texts))
                            ocr_md = reconstruct_markdown(dt_polys, rec_texts, rec_scores)
                            if has_text and len(fitz_text) > len("\n".join(rec_texts)) * 0.5:
                                all_md_parts.append(f"## Page {pn + 1}\n\n{fitz_text}\n")
                            else:
                                all_md_parts.append(f"## Page {pn + 1}\n\n{ocr_md}\n")
                        elif has_text:
                            all_results.append(f"\n--- Page {pn + 1} ---\n{fitz_text}")
                            all_md_parts.append(f"## Page {pn + 1}\n\n{fitz_text}\n")
                    elif has_text:
                        all_results.append(f"\n--- Page {pn + 1} ---\n{fitz_text}")
                        all_md_parts.append(f"## Page {pn + 1}\n\n{fitz_text}\n")

                doc.close()
            except Exception as e:
                all_results.append(f"Error: {str(e)}")
                all_md_parts.append(f"*Error: {str(e)}*")
        else:
            hdr = f"\n{'='*60}\nFile: {basename}\n{'='*60}"
            all_results.append(hdr)

            try:
                img = Image.open(file_path)
                img_w, img_h = img.size
                results = list(pipeline.predict(input=file_path))
                if results:
                    dt_polys, rec_texts, rec_scores = extract_result_texts(results[0])
                    if rec_texts:
                        all_results.append("\n".join(rec_texts))
                        md = reconstruct_markdown(dt_polys, rec_texts, rec_scores, img_w, img_h)
                        all_md_parts.append(f"# {basename}\n\n{md}")
            except Exception as e:
                all_results.append(f"Error: {str(e)}")
                all_md_parts.append(f"# {basename}\n\n*Error: {str(e)}*")

    full_text = "\n".join(all_results)
    full_md = "\n\n".join(all_md_parts)
    txt_path = save_text_file(full_text, "ocr_batch")
    md_path = save_text_file(full_md, "ocr_batch", ".md")
    return full_text, txt_path, md_path

def create_ui():
    device_status = f"GPU: {GPU_NAME}" if GPU_AVAILABLE and GPU_COUNT > 0 else "CPU only"
    if GPU_AVAILABLE and GPU_COUNT > 0:
        vram = paddle.device.cuda.get_device_properties(0).total_memory / 1024**3
        device_status += f" | VRAM: {vram:.1f}GB"

    with gr.Blocks(title="PaddleOCR GPU", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # PaddleOCR GPU Accelerated

        GPU-accelerated OCR with layout-preserving Markdown export
        **Device: {device_status}**
        """)

        with gr.Tabs():
            with gr.TabItem("Image OCR"):
                img_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Upload Image")
                        show_boxes = gr.Checkbox(label="Show detection boxes", value=True)
                        with gr.Row():
                            ocr_btn = gr.Button("Start Recognition", variant="primary", size="lg")
                            img_stop = gr.Button("Stop", variant="stop", size="lg")

                    with gr.Column(scale=1):
                        output_image = gr.Image(type="pil", label="Result Image")
                        stats_text = gr.Textbox(label="Statistics", interactive=False)
                        result_text = gr.Textbox(label="Recognized Text", lines=8, interactive=False)
                        with gr.Row():
                            dl_txt = gr.DownloadButton("Save TXT", variant="secondary", size="sm")
                            dl_md = gr.DownloadButton("Save Markdown (.md)", variant="secondary", size="sm")
                            dl_img = gr.DownloadButton("Save Image (.png)", variant="secondary", size="sm")

                def handle_image(img, boxes):
                    rimg, stats, text, txt_path, img_path, md_path = ocr_recognize(img, boxes)
                    return rimg, stats, text, txt_path or gr.skip(), img_path or gr.skip(), md_path or gr.skip()

                img_event = ocr_btn.click(
                    fn=handle_image,
                    inputs=[input_image, show_boxes],
                    outputs=[output_image, stats_text, result_text, dl_txt, dl_img, dl_md]
                )
                img_stop.click(fn=None, cancels=[img_event])

            with gr.TabItem("PDF OCR"):
                pdf_state = gr.State({})
                gr.Markdown("""
                Supports large PDFs (500MB+). Pages rendered at 200 DPI.
                **Layout preservation**: embedded text + OCR layout reconstruction → clean Markdown.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF")
                        show_boxes_pdf = gr.Checkbox(label="Show detection boxes (first 5 pages)", value=True)
                        with gr.Row():
                            pdf_btn = gr.Button("Start PDF OCR", variant="primary", size="lg")
                            pdf_stop = gr.Button("Stop", variant="stop", size="lg")

                    with gr.Column(scale=1):
                        pdf_preview = gr.Image(type="pil", label="Preview (first page)")
                        pdf_stats = gr.Textbox(label="Statistics", interactive=False)
                        pdf_text = gr.Textbox(label="Recognized Text", lines=20, interactive=False)
                        with gr.Row():
                            pdf_dl_txt = gr.DownloadButton("Save TXT", variant="secondary", size="sm")
                            pdf_dl_md = gr.DownloadButton("Save Markdown (.md)", variant="secondary", size="sm")

                def handle_pdf(pdf, boxes):
                    pimg, stats, text, txt_path, md_path = ocr_pdf(pdf, boxes)
                    return pimg, stats, text, txt_path or gr.skip(), md_path or gr.skip()

                pdf_event = pdf_btn.click(
                    fn=handle_pdf,
                    inputs=[pdf_file, show_boxes_pdf],
                    outputs=[pdf_preview, pdf_stats, pdf_text, pdf_dl_txt, pdf_dl_md]
                )
                pdf_stop.click(fn=None, cancels=[pdf_event])

            with gr.TabItem("Batch OCR"):
                gr.Markdown("Supports images (PNG/JPG/BMP) and PDF files. **Export as TXT + Markdown.**")
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(file_count="multiple", file_types=["image", ".pdf"], label="Upload Files")
                        with gr.Row():
                            batch_btn = gr.Button("Batch Recognition", variant="primary")
                            batch_stop = gr.Button("Stop", variant="stop")
                    with gr.Column(scale=1):
                        batch_result = gr.Textbox(label="Batch Results", lines=22, interactive=False)
                        with gr.Row():
                            batch_dl_txt = gr.DownloadButton("Save TXT", variant="secondary", size="sm")
                            batch_dl_md = gr.DownloadButton("Save Markdown (.md)", variant="secondary", size="sm")

                def handle_batch(files):
                    text, txt_path, md_path = batch_ocr(files)
                    return text, txt_path or gr.skip(), md_path or gr.skip()

                batch_event = batch_btn.click(
                    fn=handle_batch,
                    inputs=[batch_files],
                    outputs=[batch_result, batch_dl_txt, batch_dl_md]
                )
                batch_stop.click(fn=None, cancels=[batch_event])

        gr.Markdown(f"*Output files saved to: `{OUTPUT_DIR}`*")

    return demo

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR GPU Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PaddleOCR GPU Web UI")
    print("=" * 50)
    if GPU_AVAILABLE and GPU_COUNT > 0:
        print(f"  GPU: {GPU_NAME}")
        vram = paddle.device.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("  Running on CPU (GPU not available)")

    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"\n  Loading OCR pipeline...")
    load_result = load_pipeline()
    print(f"  {load_result}")

    if not PYMUPDF_AVAILABLE:
        print("  [WARN] PyMuPDF not installed, PDF support disabled")

    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
