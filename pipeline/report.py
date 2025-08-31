# pcb/pipeline/report.py
from pathlib import Path
import json

def build_pdf_report(metrics=None, rules=None, overlay_path=None, out_path="results/report.pdf", **kwargs):
    """
    metrics: dict  (min_*_px/mm, mm_per_px, pass, checks, violations ...)
    rules:   dict  (min_line_width_mm, min_spacing_mm, min_radius_mm, ...)
    overlay_path:  이미지 경로 (선택)
    out_path:      PDF 경로
    **kwargs:      호환성 유지를 위해 추가 인자 무시
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(out_path), pagesize=A4)
        W, H = A4

        # 제목
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20*mm, H-20*mm, "PCB/FPCB DRC Report")

        # 메트릭/규칙 요약
        c.setFont("Helvetica", 10)
        text = json.dumps({"metrics": metrics, "rules": rules}, ensure_ascii=False, indent=2)
        y = H - 30*mm
        for line in text.splitlines():
            c.drawString(20*mm, y, line[:100])
            y -= 5*mm
            if y < 35*mm:
                break

        # 오버레이 이미지(옵션)
        try:
            if overlay_path:
                img = ImageReader(str(overlay_path))
                c.drawImage(img, 20*mm, 20*mm, width=170*mm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

        c.showPage()
        c.save()
        return str(out_path)

    except Exception:
        # reportlab 미설치 등 → 텍스트로 대체
        out_txt = Path(out_path).with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(
            json.dumps({"metrics": metrics, "rules": rules, "overlay": str(overlay_path)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return str(out_txt)
