# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path
import cv2

from pipeline.preprocess import to_gray, clahe_gray, binarize, ensure_foreground_white
from pipeline.measure   import min_line_width_px, min_spacing_px
from pipeline.scale     import estimate_scale_mm_per_px
from pipeline.drc       import apply_drc
from pipeline.overlay   import make_overlay
from pipeline.skeleton  import make_skeleton, extract_polyline_points
from pipeline.curvature import min_curvature_radius_px
from pipeline.report    import build_pdf_report
from config import TARGET, CAMERA_RES, BTN_PIN, LED_PASS, LED_FAIL, BUZZ_PIN
if TARGET == "PC":
    from camera.filecam import FileCamera as CamImpl
    from io_gpio.dummy  import DummyGpio as GpioImpl
else:
    from camera.picam  import PiCamera as CamImpl
    from io_gpio.pigpio import PiGpio  as GpioImpl

app = Flask(__name__)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# 정적 파일 서빙(웹 UI에서 이미지/PDF 바로 열기용)
@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=False)

# 간단 웹 UI (templates/index.html 있으면 표시)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if not f:
        return jsonify({"status": "error", "msg": "file 파라미터가 없습니다."}), 400

    # 업로드 저장
    in_path = DATA_DIR / f.filename
    f.save(in_path)

    # 로드 & 전처리 → 이진화
    img = cv2.imread(str(in_path))
    if img is None:
        return jsonify({"status": "error", "msg": "이미지 로드 실패"}), 400
    gray   = clahe_gray(to_gray(img))
    binimg = ensure_foreground_white(binarize(gray, method="otsu", bias=0))

    # 결과 이진화 저장
    out_bin = RESULTS_DIR / "binary.jpg"
    cv2.imwrite(str(out_bin), binimg)

    # --- 측정(px) ---
    min_w_px  = round(min_line_width_px(binimg), 3)
    min_gap_px = round(min_spacing_px(binimg), 3)

    # --- 스켈레톤 & Rmin(px) ---
    skel = make_skeleton(binimg)
    poly_pts = extract_polyline_points(skel, max_points=12000)
    rmin_px = min_curvature_radius_px(poly_pts, win=9, step=2)

    # --- 스케일: 수동 우선 → 체커보드 자동 ---
    manual_mm_per_px = request.form.get("manual_mm_per_px", type=float)
    mm_per_px = manual_mm_per_px if manual_mm_per_px else estimate_scale_mm_per_px(gray)

    # --- 요구 규격(선택) ---
    req_w_mm    = request.form.get("req_w_mm", type=float)
    req_gap_mm  = request.form.get("req_gap_mm", type=float)
    req_rmin_mm = request.form.get("req_rmin_mm", type=float)

    # --- DRC 판정 ---
    drc = apply_drc(
        min_w_px, min_gap_px, mm_per_px,
        req_w_mm, req_gap_mm,
        rmin_px=rmin_px, req_rmin_mm=req_rmin_mm
    )

    # --- 오버레이 생성 ---
    overlay_path = make_overlay(
        img, binimg, min_w_px, min_gap_px, drc,
        out_path=str(RESULTS_DIR / "overlay.jpg")
    )

    # --- (옵션) PDF 리포트 생성 ---
    make_report = request.form.get("make_report", default="0")
    pdf_path = None
    if str(make_report).lower() in ("1", "true", "yes", "y"):
        pdf_path = str(RESULTS_DIR / "report.pdf")
        summary = {
            "min_line_width_px": float(min_w_px),
            "min_spacing_px":    float(min_gap_px),
            "scale_mm_per_px":   drc.get("scale_mm_per_px"),
            "min_line_width_mm": drc.get("min_line_width_mm"),
            "min_spacing_mm":    drc.get("min_spacing_mm"),
            "min_curvature_radius_px": drc.get("min_curvature_radius_px"),
            "min_curvature_radius_mm": drc.get("min_curvature_radius_mm"),
            "pass": drc.get("pass"),
        }
        imgs = {
            "source":  str(in_path),
            "binary":  str(out_bin),
            "overlay": str(overlay_path),
        }
        build_pdf_report(pdf_path, summary, imgs)

    # 웹에서 바로 열 수 있게 URL도 같이 반환
    overlay_url = f"/results/{Path(overlay_path).name}"
    binary_url  = f"/results/{out_bin.name}"
    report_url  = f"/results/{Path(pdf_path).name}" if pdf_path else None

    return jsonify({
        "status": "ok",
        "binary_path":  str(out_bin),
        "overlay_path": overlay_path,
        "report_path":  pdf_path,      # 파일 경로(서버 로컬)
        "binary_url":   binary_url,    # 브라우저에서 열 URL
        "overlay_url":  overlay_url,
        "report_url":   report_url,
        "white_ratio": float((binimg == 255).mean()),
        "min_line_width_px": float(min_w_px),
        "min_spacing_px":    float(min_gap_px),
        **drc,
        "msg": "이진화+측정(px)+Rmin+스케일/DRC+오버레이+(옵션)PDF"
    })

    def do_capture():
        cam = CamImpl() if TARGET == "PC" else CamImpl(size=CAMERA_RES)
        try:
            frame = cam.capture()
            out_path = DATA_DIR / "capture.jpg"
            cv2.imwrite(str(out_path), frame)
            return str(out_path), frame
        finally:
            cam.close()

    @app.route("/capture", methods=["POST", "GET"])
    def capture_and_analyze():
        in_path, img = do_capture()

        # 전처리 → 이진화
        gray = clahe_gray(to_gray(img))
        binimg = ensure_foreground_white(binarize(gray, method="otsu", bias=0))
        out_bin = RESULTS_DIR / "binary.jpg";
        cv2.imwrite(str(out_bin), binimg)

        # 측정(px)
        min_w_px = round(min_line_width_px(binimg), 3)
        min_gap_px = round(min_spacing_px(binimg), 3)

        # 스켈레톤 Rmin
        skel = make_skeleton(binimg)
        poly_pts = extract_polyline_points(skel, max_points=12000)
        rmin_px = min_curvature_radius_px(poly_pts, win=9, step=2)

        # 스케일·요구치(폼으로 받으면 반영)
        req = request.form if request.method == "POST" else request.args
        manual_mm_per_px = req.get("manual_mm_per_px", type=float)
        mm_per_px = manual_mm_per_px if manual_mm_per_px else estimate_scale_mm_per_px(gray)
        req_w_mm = req.get("req_w_mm", type=float)
        req_gap_mm = req.get("req_gap_mm", type=float)
        req_rmin_mm = req.get("req_rmin_mm", type=float)

        drc = apply_drc(min_w_px, min_gap_px, mm_per_px, req_w_mm, req_gap_mm,
                        rmin_px=rmin_px, req_rmin_mm=req_rmin_mm)

        overlay_path = make_overlay(img, binimg, min_w_px, min_gap_px, drc, out_path=str(RESULTS_DIR / "overlay.jpg"))

        return jsonify({
            "status": "ok", "source_path": in_path,
            "binary_path": str(out_bin), "overlay_path": overlay_path,
            "min_line_width_px": float(min_w_px), "min_spacing_px": float(min_gap_px),
            **drc
        })

    @app.route("/trigger", methods=["POST", "GET"])
    def trigger_once():
        # GPIO 준비
        gpio = GpioImpl() if TARGET == "PC" else GpioImpl(BTN_PIN, LED_PASS, LED_FAIL, BUZZ_PIN)
        try:
            ok = gpio.wait_trigger(timeout=5.0)  # GET 호출 시 5초 내 버튼 누르면 촬영
            if not ok:
                gpio.idle()
                return jsonify({"status": "idle", "msg": "no trigger"}), 200

            # 촬영
            in_path, img = do_capture()

            # 분석
            gray = clahe_gray(to_gray(img))
            binimg = ensure_foreground_white(binarize(gray, method="otsu", bias=0))
            out_bin = RESULTS_DIR / "binary.jpg";
            cv2.imwrite(str(out_bin), binimg)
            min_w_px = round(min_line_width_px(binimg), 3)
            min_gap_px = round(min_spacing_px(binimg), 3)
            skel = make_skeleton(binimg)
            poly_pts = extract_polyline_points(skel, max_points=12000)
            rmin_px = min_curvature_radius_px(poly_pts, win=9, step=2)

            # 요구치는 쿼리/폼으로
            req = request.form if request.method == "POST" else request.args
            manual_mm_per_px = req.get("manual_mm_per_px", type=float)
            mm_per_px = manual_mm_per_px if manual_mm_per_px else estimate_scale_mm_per_px(gray)
            req_w_mm = req.get("req_w_mm", type=float)
            req_gap_mm = req.get("req_gap_mm", type=float)
            req_rmin_mm = req.get("req_rmin_mm", type=float)

            drc = apply_drc(min_w_px, min_gap_px, mm_per_px, req_w_mm, req_gap_mm,
                            rmin_px=rmin_px, req_rmin_mm=req_rmin_mm)
            overlay_path = make_overlay(img, binimg, min_w_px, min_gap_px, drc,
                                        out_path=str(RESULTS_DIR / "overlay.jpg"))

            # 표시
            if drc.get("pass") is True:
                gpio.indicate_pass()
            elif drc.get("pass") is False:
                gpio.indicate_fail()
            else:
                gpio.idle()

            return jsonify({
                "status": "ok", "source_path": in_path,
                "binary_path": str(out_bin), "overlay_path": overlay_path,
                "min_line_width_px": float(min_w_px), "min_spacing_px": float(min_gap_px),
                **drc
            })
        finally:
            gpio.close()


if __name__ == "__main__":
    # 개발용 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
