# PCB/FPCB DRC with Raspberry Pi Camera

라즈베리파이 카메라로 촬영한 보드 이미지를 **원근 보정 → 벡터화 → 규칙 검사(DRC)** 까지 자동 처리하여 **선폭·간격·최소 곡률 반경(Rmin)** 위반을 검출하고, **오버레이·리포트**를 생성하는 일체형 검사 장치 프로젝트입니다.

> **One‑line mission**: 버튼 한 번 → 촬영 → 선폭/간격/곡률/비아 규칙 검사 → LED/웹UI로 결과 표시 → 리포트 자동 저장.

---

## ✨ Why (문제/가치)

* **교육·시연**: 고가 CAD/검사 장비 없이 규칙 준수 여부를 직관적으로 확인.
* **휴대성**: 전원만 꽂으면 동작하는 **일체형**(지그+조명+Pi).
* **테스트 자동화**: 버튼 트리거·판정 로깅·Pass/Fail 피드백로 **품질 게이트** 역할.

---

## 🔧 기능 개요

* **전처리**: 그레이/CLAHE, 이진화, (옵션) 투시/기울기 보정
* **피처 추출**: 스켈레톤 → 폴리라인(간소화/스무딩)
* **측정**: 최소 **선폭/간격/Rmin** (px, mm)
* **단위 보정**: 체커보드/피두셜 기반 **px→mm** (수동 2점 캘리브레이션 대안)
* **DRC 검사**: min line/space/radius, (옵션) **플렉스 영역 내 비아 금지**
* **결과 출력**: 오버레이(PNG), 리포트(PDF/txt), JSON
* **하드웨어 연동**: 버튼 입력 → 촬영/분석/LED·부저(옵션) 피드백

---

## 📦 프로젝트 구조 (권장)

```
project_root/
├─ app.py                      # Flask 서버(라즈베리파이용; PC는 선택)
├─ cli/
│  └─ analyze_file.py          # ★ 단일 이미지 CLI 분석 도구
├─ pipeline/
│  ├─ preprocess.py            # to_gray/clahe/binarize/deskew 등
│  ├─ skeleton.py              # make_skeleton / extract_polyline_points
│  ├─ measure.py               # min_line_width_px / min_spacing_px
│  ├─ curvature.py             # min_curvature_radius_px
│  ├─ scale.py                 # estimate_scale_mm_per_px
│  ├─ drc.py                   # apply_drc
│  └─ report.py                # build_pdf_report
├─ tools/
│  ├─ calibrate_two_point.py   # 두 점 클릭 → mm_per_px JSON
│  └─ make_flex_roi.py         # 플렉스 영역 다각형(ROI) 생성 도구
├─ calibration/
│  └─ mm_per_px.json           # 수동 캘리브레이션 결과(선택)
├─ data/
│  └─ pcb2.jpg                 # 샘플 이미지
├─ results_cli/                # CLI 결과물 출력 폴더
├─ rules.json                  # DRC 규칙/ROI/VIA 파라미터
└─ venv/ ...                   # 가상환경(플랫폼별)
```

---

## 🖥️ 설치

* Python **3.10+** 권장
* 의존성:

```bash
pip install numpy opencv-py
``