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

## 개발 진행 기록 및 전체 데이터 학습 절차 (2026-10-15)

### 지금까지 완료된 사항
- GitHub AI/Backend/Frontend 저장소 및 각 README 생성.
- 원본 PSV 저장 경로 확인 (전체 파일 무결성 확인은 별도 필요).
- XGBoost smoke test **사용자가 로컬에서 실행 성공**: 200명, Train 128/Val 32/Test 40명, 학습 샘플 2,229(양성 71), Validation AUROC **0.6532**, AUPRC **0.4857**, 양성 시점 비율 **0.0489**. 소규모 표본의 결과이며 최종 성능으로 인용하지 말 것.
- `src/preprocessing/prepare_data.py`: Set A/B 환자별 파일 읽기, 결측률·측정 빈도·평균·양성률 등 기초 EDA, 출처와 환자 양성 여부를 고려한 환자 단위 분할 생성.
- `src/training/train_xgboost.py`: 기존 160개 특징을 유지하면서 재현 가능한 무작위 smoke test와 분할 매니페스트 입력 지원.

### 팀원이 처음 실행하는 순서 (프로젝트 루트 기준)
```bash
python -m pip install -r requirements.txt
# 먼저 빠른 데이터 비교 및 분할 점검 (소규모 표본)
python src/preprocessing/prepare_data.py --max-patients 2000
# 소규모 분할표에 따라 실제 XGBoost 학습
python src/training/train_xgboost.py --split-manifest data/splits/patient_splits_sample2000.json

# 전체 환자 EDA 및 환자 단위 64/16/20 분할 (시간·메모리 필요)
python src/preprocessing/prepare_data.py
# 전체 데이터 XGBoost 베이스라인 학습 (수십 분 이상 걸릴 수 있음)
python src/training/train_xgboost.py --split-manifest data/splits/patient_splits.json
```

- `prepare_data.py`는 소규모/전체 실행마다 `data/splits/patient_index.csv`를 덮어쓰므로 **반드시 분할표와 같은 실행의 인덱스 파일을 유지**해야 함.
- EDA 출력: `reports/set_ab_eda_sample2000.csv` 또는 `reports/set_ab_eda.csv`; 전체 Set A/B의 차이를 검토한 뒤 병합 정책 확정.
- `data/splits/patient_splits*.json`: 출처(Set A/B) 및 환자 양성 여부 층화, `seed=42`, 환자 단위 비중 Train 64%, Validation 16%, Test 20%.
- 24시간 미만인 환자는 현 XGBoost 코드에서 학습용 window를 생성하지 않음. 이는 향후 초기 시간 예측 정책과 함께 검토해야 함.
- `SepsisLabel`은 이미 6시간 조기경보 라벨이므로 시점 이동 금지. 특징 생성 시 현재 시점 이후 데이터 사용 금지.
- 전처리 전략 중 과제 필수 Forward Fill / Interpolation / KNN Imputation 적용과 비교는 **추후 구현**; 현 코드는 관측값만으로 통계를 계산하고 XGBoost가 결측 특징을 자체 처리.
- 파일 출처에 따른 분할은 정확도를 보장하지 않음. 임상적 분포 차이가 크면 별도의 Set A→B 일반화 실험 고려.

### 다음 개발 단계
1. 전체 PSV 파일 무결성 및 Set A/B EDA 검증, 관찰 기간/결측/라벨 정책 확정
2. 미래 데이터 누수 없는 Forward Fill 등 적용, 특징/원시 시퀀스 공통 파이프라인
3. XGBoost 전체 학습 + 5-Fold Group K-Fold (개발 세트만), 검증 성능 안정성 측정
4. 공식 PhysioNet Utility Score, 혼동행렬/PR Curve, 임계값 및 경보 피로 분석
5. Platt/Isotonic Calibration 및 Brier/ECE, SHAP
6. LSTM/GRU, Soft Voting 앙상블, 실험 기록 MLflow/WandB
7. 가능한 SOFA/qSOFA 계산 범위 명시, 불가능한 항목 데이터 한계 리포트
8. FastAPI 추론 계약과 Streamlit 통합, Docker Compose, 지연·재현성 테스트, 기술 보고서

> 현재 모델 성능 목표 달성 여부는 미확정이다. 전체 데이터 학습, 교차검증, 공식 Utility Score, 보정, 서비스 구현은 아직 완료되지 않았다.

## AI 개발 진행 현황 (2026-11-08, GitHub 원격 확인 기준)

- 고정 코호트: Train 25,814명 / Validation 6,454명 / Test 8,068명. 동일 환자의 시간별 행이 분할 사이에 중복되지 않도록 관리.
- XGBoost 176 특징: Validation AUROC 0.790062, AP 0.094880.
- 최종 LSTM (`pos_weight=10`, 24×64, Epoch 6): Validation AUROC 0.829429, AP 0.135410.
- 25:75 XGBoost-LSTM 앙상블: Validation AUROC 0.834336, AP 0.138519. 환자 부트스트랩에서 LSTM 대비 개선이 확실하다고 보기 어려워 최종 Test 모델로 선택하지 않음.
- **잠금 LSTM 최초 Test**: 예측 가능 환자 6,149명, 예측 가능 시점 135,693개. AUROC **0.824396**, AP **0.131886**, Brier **0.052583**, ECE(10-bin) **0.114430**, 전체 Test 환자 시간축 normalized utility **0.297674**. Test 환자 전체 8,068명. 최초 23개 시점 및 24행 미만 환자의 경보를 0으로 두고, 임계값 **0.2**, 별도 calibration 및 경보 cooldown 없음.
- 최종 Test는 **다시 실행하거나 이 결과로 모델/임계값을 최적화하지 않음**. AUROC 0.82 목표 달성, AP 0.40 목표 미달. 이는 임상 검증 결과가 아닌 연구용 고정 평가임.
- 설명/검증: XGBoost Native TreeSHAP와 그림, Validation PR Curve/Reliability Diagram, Sensitivity@≥95% Specificity, 공식 Utility 분기 독립 패리티 테스트 1,250개 사례, Calibration 진단 완료.
- LSTM 5-Fold CV는 **원래 Train 환자 내부**에서 진행 중. GitHub에 Fold 1/2의 학습 기록이 반영되어 있고 Fold 3/4/5 완료 여부는 로컬 결과 확인 필요. 각 Fold는 별도의 통계·체크포인트를 사용하고 원래 Validation·Test에 접근하지 않음.
- 최종 모델 파일 및 최종 Test JSON은 로컬 산출물 관리 대상이다. 원격에 파일이 보이지 않는다고 학습이나 평가가 미실행된 것은 아니므로 원격 확인 사실과 로컬 실행 로그를 구분해야 한다.

### 남은 LSTM CV 단계 (로컬 PowerShell, 프로젝트 루트)

기존 Fold 3 학습이 완료되었는지 먼저 확인한 후, 미완료 상태라면 기존 체크포인트/기록을 확인하고 중복 학습으로 덮어쓰지 않도록 한다.

```powershell
git pull origin main
Get-Content reports/lstm_full_pw10_fold3_history.json -Tail 18
Test-Path models/lstm_full_pw10_fold3_best.pt
# Fold 3이 최종 완료된 경우에만 다음 단계 진행
python src/training/prepare_lstm_group_cv.py --fit-stats-fold 4
python src/training/train_lstm.py --manifest data/splits/lstm_cv/fold4.json --stats models/lstm_cv_fold4_stats.json --run-tag fold4 --epochs 12 --patience 3 --pos-weight 10
python src/training/prepare_lstm_group_cv.py --fit-stats-fold 5
python src/training/train_lstm.py --manifest data/splits/lstm_cv/fold5.json --stats models/lstm_cv_fold5_stats.json --run-tag fold5 --epochs 12 --patience 3 --pos-weight 10
# 전 Fold의 학습 완료 및 모델 파일 존재가 확인된 뒤에만 실행
python src/training/summarize_lstm_group_cv.py
```

5-Fold 결과 집계는 각 Fold에서 AP가 가장 높은 Epoch의 AUROC/AP로 평균 및 표본 표준편차를 구한다. 동일 Fold의 최적 Epoch 선택에 사용된 점수이므로 독립적인 임상 일반화 성능으로 해석하지 않는다. 기존 Notion AI 개발계획서 및 완료한 Test 잠금 정책은 유지한다.
