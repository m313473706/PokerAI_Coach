from __future__ import annotations
import argparse, json, platform, sys, time, re
from pathlib import Path
import cv2

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
DEFAULT_IMAGE = ROOT / "data" / "ocr_smoke" / "action_all_in_label.png"


def run_smoke(image_path: Path, require_action_text: bool = False) -> tuple[dict, int]:
    report = {
        "test": "RapidOCR action-label runtime smoke",
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "frozen_exe": bool(getattr(sys, "frozen", False)),
        "image_path": str(image_path),
        "image_loaded": False,
        "rapidocr_imported": False,
        "engine_initialized": False,
        "inference_completed": False,
        "recognized_texts": [],
        "expected_action": "all_in",
        "expected_action_detected": False,
        "require_action_text": bool(require_action_text),
        "runtime_error": None,
        "warning": None,
    }
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Smoke image not found/readable: {image_path}")
        report["image_loaded"] = True
        if img.shape[1] < 300:
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        from rapidocr_onnxruntime import RapidOCR
        report["rapidocr_imported"] = True
        engine = RapidOCR()
        report["engine_initialized"] = True
        t0 = time.perf_counter()
        result, elapsed = engine(img)
        report["inference_completed"] = True
        report["elapsed_s"] = round(float(time.perf_counter() - t0), 3)
        report["engine_elapsed"] = str(elapsed)
        texts = []
        for row in result or []:
            try:
                text = str(row[1]).strip()
                score = float(row[2])
                if text:
                    texts.append({"text": text, "score": round(score, 3)})
            except Exception:
                continue
        report["recognized_texts"] = texts
        normalized = "".join(re.sub(r"\s+", "", t["text"]) for t in texts)
        report["expected_action_detected"] = ("全押" in normalized) or ("全下" in normalized) or ("ALLIN" in normalized.upper()) or ("ALL-IN" in normalized.upper())
        if not report["expected_action_detected"]:
            report["warning"] = "RapidOCR loaded and inferred successfully, but this smoke crop did not yield the expected all-in text."
            if require_action_text:
                report["runtime_error"] = "Strict action-text check failed: expected all-in label was not detected."
                return report, 3
        return report, 0
    except Exception as exc:
        report["runtime_error"] = f"{type(exc).__name__}: {exc}"
        return report, 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-smoke", action="store_true", help="Run bundled RapidOCR initialize+inference smoke")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--require-action-text", action="store_true", help="Fail unless the sample all-in text is actually recognized")
    args = parser.parse_args(argv)
    report, code = run_smoke(args.image, require_action_text=args.require_action_text)
    if args.report:
        out = args.report
    elif getattr(sys, "frozen", False):
        out = Path.home() / "Documents" / "PokerAI_Recognition_Output" / "ocr_runtime_smoke.json"
    else:
        out = Path(__file__).resolve().parent / "reports" / "ocr_runtime_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows GitHub runners may use a legacy console code page (for example cp1252).
    # Keep console output ASCII-only; the full Unicode report is already written as UTF-8 JSON.
    console = getattr(sys, "stdout", None)
    if console is not None:
        console_report = json.dumps(report, ensure_ascii=True, indent=2)
        for line in console_report.splitlines():
            _safe_console_line(line)
        _safe_console_line(f"OCR smoke report written: {out.as_posix()}")
    return code


def _safe_console_line(text: str) -> None:
    """Write a log line without crashing on Windows legacy encodings or windowed EXEs."""
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except UnicodeEncodeError:
        # Backslash escaping preserves diagnosability without depending on a code page.
        safe = text.encode("ascii", errors="backslashreplace").decode("ascii")
        try:
            stream.write(safe + "\n")
            stream.flush()
        except Exception:
            pass
    except Exception:
        # A --windowed PyInstaller executable may have no attached console.
        # The JSON report file remains the source of truth.
        pass

if __name__ == "__main__":
    sys.exit(main())
