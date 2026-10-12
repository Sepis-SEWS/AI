# SEWS AI

PhysioNet Challenge 2019 기반 ICU 패혈증 조기 경보 시스템의 데이터 전처리, 특징 추출, 모델 학습 및 추론 모듈을 개발하는 저장소입니다.

## Directory layout

- `data/raw/`: 원본 PSV 데이터 (Git 제외)
- `data/splits/`: 환자 단위 Train/Validation/Test 분할 정보
- `data/processed/`: 전처리된 데이터 (Git 제외)
- `src/preprocessing/`: 전처리
- `src/features/`: 시계열 특징 추출
- `src/training/`: XGBoost, LSTM 및 앙상블 학습
- `src/evaluation/`: 성능 평가 및 Calibration
- `src/inference/`: 모델 추론
- `models/`: 학습 모델 (Git 제외)
- `configs/`: 설정
- `reports/`: 실험 보고서
- `tests/`: 테스트

원본 데이터와 모델 가중치는 Git에 커밋하지 않습니다.
