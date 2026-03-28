# YOLOv11 Edge-AI Benchmark (Jetson Orin Nano)
> **NVIDIA Jetson 환경에서 YOLOv11 모델의 성능을 측정하고 최적화하는 프로젝트**


---

## 환경 설정 (Environment)
* **Device:** NVIDIA Jetson Orin Nano (8GB)
* **HOST OS:** L4T R36.4.3 (JetPack 6.2)
* **Model:** YOLOv11n


자세한 설정 방법은
##### https://velog.io/@cint/Edge-AI-Jetson-Orin-Nano-Super-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EA%B0%9D%EC%B2%B4-%ED%83%90%EC%A7%80-%EB%B0%8F-%EC%B5%9C%EC%A0%81%ED%99%94-1-%EC%84%B8%ED%8C%85-%EB%B0%8F-Baseline

---

## 데이터 다운로드

mAP를 계산하기 위해서 **COCO 2017 validation** 데이터셋을 사용

데이터셋 다운로드 링크 :

labels & annotations : https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip

images : http://images.cocodataset.org/zips/val2017.zip

YOLO 라벨은 아래 문서를 참고:

https://docs.ultralytics.com/ko/datasets/detect/coco/#dataset-structure

**데이터 폴더 구조**
```
data/
├── images/
│   └── val2017/
│       ├── 000000000139.jpg
│       └── ...
│
├── labels/
│   └── val2017/
│       ├── 000000000139.txt
│       └── ...
│
└── instances_val2017.json
```

---

## 사용법 (Usage)

```bash
python main.py [OPTIONS]
```

---

## Arguments

| Argument | Type | Default | Description |
|--------|------|--------|-------------|
| `--device` | str | `cpu` | 실행 디바이스 (`cpu` 또는 `cuda`) |
| `--show` | flag | False | 영상 출력 창 표시 |
| `--nosave` | flag | False | 결과 영상 저장 비활성화 |
| `--mode` | str | `run` | 실행 모드 |
| `--frames` | int | 1000 | 처리할 프레임 수 |
| `--camera` | int | None | 카메라 입력 사용 (예: 0) |

---

## Modes

### 1. `run`
기본 실행 모드 (추론 수행)

```bash
python main.py --mode run
```

---

### 2. `latency`
latency benchmark 측정

```bash
python main.py --mode latency
```

---

### 3. `hardware`
RAM 사용량 등 하드웨어 성능 측정

```bash
python main.py --mode hardware
```

---

### 4. `map`
mAP (mean Average Precision) 평가

```bash
python main.py --mode map
```

- IoU threshold: `0.5 ~ 0.95` (0.05 간격)
- confidence threshold: `0.001`

---

##  Input Source

###  Video 파일 (기본)

```bash
python main.py
```

- 기본 영상: `17431598-hd_1920_1080_60fps.mp4`

---

###  Camera 사용

```bash
python main.py --camera 0
```

- `0`: 기본 웹캠

---

##  Example Commands

### CUDA로 실행 + 화면 출력

```bash
python main.py --device cuda --show
```

---

### Latency 측정 (500 프레임)

```bash
python main.py --mode latency --frames 500
```

---

### Camera 입력 + 저장 안함

```bash
python main.py --camera 0 --nosave
```
