### psv파일이란?

CSV와 데이터 구조 자체는 동일하나 구분자만 다른 차이를 가진 표형식 파일임.

### 각 psv 정보

psv 파일 하나는 환자 한 명에 대응함. 각 폴더 별 총합 4만 명 규모의 시계열 임상 데이터임.
psv 파일을 정리한 내용은 아래와 같음.

#### 전체 컬럼 수

총 41개 컬럼으로 구성됨.

#### 컬럼 순서 및 의미 (원본 헤더 순서 그대로)

아래 "의미"는 더 이상 추정이 아니라, PhysioNet/Computing in Cardiology Challenge 2019 공식 매뉴스크립트(`physionet_challenge_2019_ccm_manuscript.pdf`, Reyna et al., 2019) **Table 1**에 적힌 공식 정의를 그대로 번역한 것임. (출처: `/Users/looksh/Downloads/DESTINATION/physionet_challenge_2019_ccm_manuscript.pdf`, 7페이지)

| 순서 | 컬럼명           | 의미 (공식 정의, Table 1 기준)                                                |
| ---- | ---------------- | ------------------------------------------------------------------------------ |
| 1    | HR               | 심박수 (beats/min)                                                              |
| 2    | O2Sat            | 맥박산소측정법 산소포화도 (%)                                                   |
| 3    | Temp             | 체온 (°C)                                                                       |
| 4    | SBP              | 수축기 혈압 (mmHg)                                                              |
| 5    | MAP              | 평균 동맥압 (mmHg)                                                              |
| 6    | DBP              | 이완기 혈압 (mmHg)                                                              |
| 7    | Resp             | 호흡수 (breaths/min)                                                            |
| 8    | EtCO2            | 호기말 이산화탄소분압 (mmHg)                                                    |
| 9    | BaseExcess       | 초과 중탄산염 (mmol/L)                                                          |
| 10   | HCO3             | 중탄산염 (mmol/L)                                                               |
| 11   | FiO2             | 흡입산소농도 (%)                                                                |
| 12   | pH               | 혈액 산도(pH)                                                                   |
| 13   | PaCO2            | 동맥혈 이산화탄소분압 (mmHg)                                                    |
| 14   | SaO2             | 동맥혈 산소포화도 (%)                                                           |
| 15   | AST              | 아스파르테이트아미노전이효소 (IU/L, 간수치)                                     |
| 16   | BUN              | 혈중요소질소 (mg/dL, 신장수치)                                                  |
| 17   | Alkalinephos     | 알칼리인산분해효소 (IU/L, 간수치)                                               |
| 18   | Calcium          | 칼슘 (mg/dL)                                                                    |
| 19   | Chloride         | 염소 (mmol/L)                                                                   |
| 20   | Creatinine       | 크레아티닌 (mg/dL, 신장수치)                                                    |
| 21   | Bilirubin_direct | 직접 빌리루빈 (mg/dL, 간수치)                                                   |
| 22   | Glucose          | 혈청 포도당 (mg/dL)                                                             |
| 23   | Lactate          | 젖산 (mg/dL)                                                                    |
| 24   | Magnesium        | 마그네슘 (mmol/dL)                                                              |
| 25   | Phosphate        | 인 (mg/dL)                                                                      |
| 26   | Potassium        | 칼륨 (mmol/L)                                                                   |
| 27   | Bilirubin_total  | 총 빌리루빈 (mg/dL)                                                             |
| 28   | TroponinI        | 트로포닌I (ng/mL, 심근효소)                                                     |
| 29   | Hct              | 헤마토크릿 (%)                                                                  |
| 30   | Hgb              | 헤모글로빈 (g/dL)                                                               |
| 31   | PTT              | 부분트롬보플라스틴시간 (seconds, 응고 수치)                                     |
| 32   | WBC              | 백혈구 수 (count/L)                                                             |
| 33   | Fibrinogen       | 피브리노겐 농도 (mg/dL)                                                         |
| 34   | Platelets        | 혈소판 수 (count/mL)                                                            |
| 35   | Age              | 나이 (years)                                                                    |
| 36   | Gender           | 성별: 여성(0) / 남성(1)                                                          |
| 37   | Unit1            | 입실 병동이 MICU(내과계 중환자실)인지 여부: 아니오(0) / 예(1)                    |
| 38   | Unit2            | 입실 병동이 SICU(외과계 중환자실)인지 여부: 아니오(0) / 예(1)                    |
| 39   | HospAdmTime      | 병원 입원 시점부터 ICU 입실까지 걸린 시간 (ICU 입실 시점 기준 시간, hours)       |
| 40   | ICULOS           | ICU 재원 시간 (ICU 입실 후 경과 시간, hours)                                    |
| 41   | SepsisLabel      | 패혈증 환자는 t ≥ t_sepsis − 6 이면 1, 아니면 0. 비패혈증 환자는 항상 0          |

`SepsisLabel`은 "그 시점에 이미 패혈증 진단 기준을 충족했는지"가 아니라, **패혈증 발병 시점(t_sepsis)보다 6시간 이전부터 1로 표시되는 조기 예측용 레이블**임. 즉 실제 발병 6시간 전부터 모델이 1을 예측하도록 설계된 것.

#### 컬럼 카테고리 분류

공식 매뉴스크립트 Table 1의 분류 방식 그대로임: 생체징후(1~8행), 혈액검사(9~34행), 인구통계(35~40행), 결과 레이블(41행). 한 행 = 컬럼 1개 (압축 없음).

**1) 생체징후 (Vital signs) — 8개, 매시간 자동 측정되는 편**

| 컬럼명 | 의미                  |
| ------ | --------------------- |
| HR     | 심박수                |
| O2Sat  | 산소포화도            |
| Temp   | 체온                  |
| SBP    | 수축기 혈압           |
| MAP    | 평균 동맥압           |
| DBP    | 이완기 혈압           |
| Resp   | 호흡수                |
| EtCO2  | 호기말 이산화탄소분압 |

**2) 혈액검사 (Lab values) — 26개, 채혈 시점에만 기록되어 대부분 NaN**

| 컬럼명           | 의미                                  |
| ---------------- | ------------------------------------- |
| BaseExcess       | 초과 중탄산염                         |
| HCO3             | 중탄산염                              |
| FiO2             | 흡입산소농도                          |
| pH               | 혈액 산도                             |
| PaCO2            | 동맥혈 이산화탄소분압                 |
| SaO2             | 동맥혈 산소포화도                     |
| AST              | 아스파르테이트아미노전이효소 (간수치) |
| BUN              | 혈중요소질소 (신장수치)               |
| Alkalinephos     | 알칼리인산분해효소 (간수치)           |
| Calcium          | 칼슘                                  |
| Chloride         | 염소                                  |
| Creatinine       | 크레아티닌 (신장수치)                 |
| Bilirubin_direct | 직접 빌리루빈 (간수치)                |
| Glucose          | 혈청 포도당                           |
| Lactate          | 젖산                                  |
| Magnesium        | 마그네슘                              |
| Phosphate        | 인                                    |
| Potassium        | 칼륨                                  |
| Bilirubin_total  | 총 빌리루빈                           |
| TroponinI        | 트로포닌I (심근효소)                  |
| Hct              | 헤마토크릿                            |
| Hgb              | 헤모글로빈                            |
| PTT              | 부분트롬보플라스틴시간 (응고 수치)    |
| WBC              | 백혈구 수                             |
| Fibrinogen       | 피브리노겐 농도                       |
| Platelets        | 혈소판 수                             |

**3) 인구통계 (Demographics) — 6개, 공식 분류 기준으로는 시간 관련 2개(HospAdmTime, ICULOS)도 여기 포함됨**

| 컬럼명      | 의미                                            |
| ----------- | ----------------------------------------------- |
| Age         | 나이                                             |
| Gender      | 성별 (여성 0 / 남성 1)                          |
| Unit1       | MICU(내과계 중환자실) 입실 여부 (0/1)           |
| Unit2       | SICU(외과계 중환자실) 입실 여부 (0/1)           |
| HospAdmTime | 병원 입원 ~ ICU 입실까지 걸린 시간              |
| ICULOS      | ICU 입실 후 경과 시간                           |

**4) 결과 레이블 (Outcome) — 1개**

| 컬럼명      | 의미                                                        |
| ----------- | ------------------------------------------------------------ |
| SepsisLabel | 패혈증 조기 예측 레이블 (t ≥ t_sepsis−6 이면 1, 그 외 0)     |

카테고리별 개수 합: 8 + 26 + 6 + 1 = **41개** (전체 컬럼 수와 일치)

#### 환자 1 (p000001.psv) 시간대별 예시 (41개 컬럼 전체)

41개 컬럼을 모두 가로로 나열하면 표가 너무 길어져서 읽기 어려워지므로, 컬럼명을 행으로, 시점(ICULOS)을 열로 바꿔서(전치, transpose) 정리함. 컬럼은 41개 전부 포함했고, 시점(행 방향이었던 54개)은 값이 채워진 시점 위주로 일부만 발췌함. 이 환자는 마지막 시점(ICULOS=54)까지 SepsisLabel이 계속 0으로, 관찰 기간 내 패혈증 기준을 충족하지 않음.

| 컬럼명           | ICULOS=1 | ICULOS=2 | ICULOS=3 | ... | ICULOS=17 | ... | ICULOS=40 | ... | ICULOS=54 |
| ---------------- | -------- | -------- | -------- | --- | --------- | --- | --------- | --- | --------- |
| HR               | NaN      | 97       | 89       | ... | 103       | ... | 117       | ... | 84        |
| O2Sat            | NaN      | 95       | 99       | ... | 90        | ... | NaN       | ... | 85        |
| Temp             | NaN      | NaN      | NaN      | ... | NaN       | ... | 36.78     | ... | NaN       |
| SBP              | NaN      | 98       | 122      | ... | NaN       | ... | 126       | ... | 78        |
| MAP              | NaN      | 75.33    | 86       | ... | NaN       | ... | 104.67    | ... | 44        |
| DBP              | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| Resp             | NaN      | 19       | 22       | ... | 30        | ... | 22        | ... | 18        |
| EtCO2            | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| BaseExcess       | NaN      | NaN      | NaN      | ... | 21        | ... | NaN       | ... | NaN       |
| HCO3             | NaN      | NaN      | NaN      | ... | 45        | ... | 48        | ... | NaN       |
| FiO2             | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| pH               | NaN      | NaN      | NaN      | ... | 7.37      | ... | NaN       | ... | NaN       |
| PaCO2            | NaN      | NaN      | NaN      | ... | 90        | ... | NaN       | ... | NaN       |
| SaO2             | NaN      | NaN      | NaN      | ... | 91        | ... | NaN       | ... | NaN       |
| AST              | NaN      | NaN      | NaN      | ... | 16        | ... | NaN       | ... | NaN       |
| BUN              | NaN      | NaN      | NaN      | ... | 14        | ... | 22        | ... | NaN       |
| Alkalinephos     | NaN      | NaN      | NaN      | ... | 98        | ... | NaN       | ... | NaN       |
| Calcium          | NaN      | NaN      | NaN      | ... | 9.3       | ... | 9.6       | ... | NaN       |
| Chloride         | NaN      | NaN      | NaN      | ... | 85        | ... | 85        | ... | NaN       |
| Creatinine       | NaN      | NaN      | NaN      | ... | 0.7       | ... | 0.7       | ... | NaN       |
| Bilirubin_direct | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| Glucose          | NaN      | NaN      | NaN      | ... | 193       | ... | 133       | ... | NaN       |
| Lactate          | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| Magnesium        | NaN      | NaN      | NaN      | ... | 2         | ... | 2.2       | ... | NaN       |
| Phosphate        | NaN      | NaN      | NaN      | ... | 3.3       | ... | 3.7       | ... | NaN       |
| Potassium        | NaN      | NaN      | NaN      | ... | 3.8       | ... | 4.6       | ... | NaN       |
| Bilirubin_total  | NaN      | NaN      | NaN      | ... | 0.3       | ... | NaN       | ... | NaN       |
| TroponinI        | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| Hct              | NaN      | NaN      | NaN      | ... | 37.2      | ... | 36.2      | ... | NaN       |
| Hgb              | NaN      | NaN      | NaN      | ... | 12.5      | ... | 12.2      | ... | NaN       |
| PTT              | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| WBC              | NaN      | NaN      | NaN      | ... | 5.7       | ... | 14.7      | ... | NaN       |
| Fibrinogen       | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| Platelets        | NaN      | NaN      | NaN      | ... | 317       | ... | 338       | ... | NaN       |
| Age              | 83.14    | 83.14    | 83.14    | ... | 83.14     | ... | 83.14     | ... | 83.14     |
| Gender           | 0        | 0        | 0        | ... | 0         | ... | 0         | ... | 0         |
| Unit1            | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| Unit2            | NaN      | NaN      | NaN      | ... | NaN       | ... | NaN       | ... | NaN       |
| HospAdmTime      | -0.03    | -0.03    | -0.03    | ... | -0.03     | ... | -0.03     | ... | -0.03     |
| SepsisLabel      | 0        | 0        | 0        | ... | 0         | ... | 0         | ... | 0         |
