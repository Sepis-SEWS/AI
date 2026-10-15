# SEWS — ICU 패혈증 조기 경보 시스템

PhysioNet/CinC Challenge 2019의 ICU 환자 시간별 기록으로 **최근 24시간 관찰 → 패혈증 약 6시간 조기경보**를 구현하는 팀 프로젝트입니다. 과제/연구용 프로토타입으로, 실제 임상 적용이 검증된 의료기기가 아닙니다.

## 레포지토리 구성
- [AI](https://github.com/Sepis-SEWS/AI): 데이터·특징·XGBoost/LSTM·앙상블·검증·SHAP·Calibration
- [Backend](https://github.com/Sepis-SEWS/Backend): FastAPI 모델 서빙 및 예측 API
- [Frontend](https://github.com/Sepis-SEWS/Frontend): Streamlit 임상 모니터링 화면

## 공통 원칙
- 공식 `SepsisLabel`은 이미 발병 6시간 전에 양성으로 전환된 라벨이므로 **추가로 6시간 이동하지 않음**.
- 환자별 Train/Validation/Test 분리, 과거부터 현재까지 데이터만 사용, 평가 데이터로 전처리 통계 학습 금지.
- Streamlit은 AI 모델 파일을 직접 읽지 않음. 반드시 FastAPI를 호출.
- 성능 목표(미달성): AUROC ≥0.82, AUPRC ≥0.40, Brier ≤0.15, CPU 4코어 추론 평균 ≤200ms.
- 실험 결과 없이 달성 여부를 주장하지 않음. 실제 측정값과 예시를 구분.
- 환경: Python 3.9~3.11, PyTorch ≥2.0 또는 TensorFlow ≥2.10, Docker Compose, Ubuntu 22.04/WSL2 허용.
- 유형힌트·docstring·Conventional Commits·고정 라이브러리 버전 적용.

## API 계약 초안 (개발 전 협의 필요)
요청: 환자 식별자 + 현재 시점까지의 최근 24시간 기록(JSON). 응답: 위험 점수(`risk_score`), Calibration 점수(`calibrated_probability`), 위험 단계(`risk_level`), 계산 가능한 SOFA 요소와 SHAP 설명. 누락 필드는 null 또는 '계산 불가' 명시. `Warning 0.5~0.75`, `Alert ≥0.75`는 과제 예시이며 실제 임계값은 검증 데이터로 확정.

## 구현 주의
과제에 기재된 활력징후 8개/검사 10개 중 원본 칼럼명과 불일치하는 항목(RR·SpO2·INR·PaO2/FiO2)이 있음. 팀 계획의 원본 실제 변수는 활력징후 7개 + 검사 9개(총 16개). 누락 원천정보(GCS, 승압제, 소변량 등) 없이 완전한 SOFA/qSOFA 산출 불가. 대체 계산/미지원 상태를 명확히 기록해야 함.

## AI 영역 요구사항
1. Set A/B 비교 EDA: 결측률, 측정 빈도, 변수 분포, 양성률 및 병합 근거.
2. 환자 단위 분할 고정; 24시간 causal sliding window와 환자별 시점 라벨.
3. 16개 원본 변수 × 10개 통계 특징(Mean, Std, Min, Max, Median, IQR, Skewness, Slope, Rate of Change, CV) — 최대 160개 계획. 결측치로 계산 불가한 통계는 안전하게 처리.
4. XGBoost/LightGBM 베이스라인, LSTM/GRU, Soft Voting/Stacking; 클래스 불균형 대응.
5. 5-Fold Group KFold; AUROC/AUPRC/PR Curve/Sensitivity@95% Specificity/Confusion Matrix/공식 Utility Score.
6. Calibration Curve, Brier, ECE 및 Platt/Isotonic; SHAP; MLflow/WandB 실험 추적.
7. 학습 결과 및 전처리 계약 버전 관리, 추론 코드 분리. 실제 수치는 검증 후 기록.

## 현재 데이터 및 폴더
원본 PSV 경로: `data/data-sepsis/training/training_setA/`, `training_setB/`. 실제 데이터 업로드는 별도 관리하고 공개/용량 및 배포 조건을 점검해야 함.

- `src/preprocessing/`: 환자별 로드, 결측/누수 검증, 그룹 분할
- `src/features/`: 24시간 통계 특징과 LSTM용 원시 시퀀스
- `src/training/`: XGBoost/LSTM/앙상블 학습
- `src/evaluation/`: 지표, Calibration, Utility
- `src/inference/`: 학습/서빙 공유 전처리 및 추론
- `data/splits/`: 환자 분할 목록
- `data/processed/`: 로컬 특징 행렬
- `models/`: 로컬 모델 산출물
- `configs/`, `reports/`, `tests/`: 설정·보고서·검증

## 첫 실행 순서
```bash
python -m pip install -r requirements.txt
python src/training/train_xgboost.py --max-patients 200
```
빠른 기능 확인용이며 성능 평가로 사용하지 않음. 이후 전체 환자에 대해 실행하고 환자 단위 5-Fold/Utility/Calibration을 추가해야 함. 현재 원본 파일 개수·무결성 전체 검증은 미완료.

## 팀 역할 및 개인 기여
추후 팀원별 담당/실제 커밋·실험을 기록할 것.
