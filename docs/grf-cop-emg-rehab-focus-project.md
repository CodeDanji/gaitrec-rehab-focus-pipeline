# GRF/COP + EMG 기반 재활 초점 제안 프로젝트 브리프

작성일: 2026-05-20

## 1. 프로젝트 한 줄 요약

보행 중 지면반력(GRF), 압력중심(COP), 보조 EMG 정보를 활용해 근골격계 손상 위치와 연관된 보행 기능 이상을 분석하고, 특정 근육의 원인을 단정하기보다 재활에서 우선 확인해야 할 관절/기능 후보를 제안한다.

## 2. 문제의식

병원 진료는 종종 "허리 디스크가 있다", "무릎이 안 좋다", "고관절에 문제가 있다"처럼 결과 중심으로 설명된다. 하지만 실제 재활에서는 특정 부위의 통증이나 손상이 그 부위만의 문제가 아닐 수 있다. 몸은 연결되어 있고, 어느 부위의 가동성 제한, 약화, 체중부하 회피, 보상 움직임이 다른 부위의 과사용이나 부하 증가로 이어질 수 있다.

이 프로젝트는 질병명을 맞히는 AI가 아니라, 보행 데이터에서 기능적 이상을 찾고 "재활에서 무엇을 더 확인해야 하는가"를 제안하는 설명 가능한 분석을 목표로 한다.

## 3. 주요 생체 신호 개념

- GRF(Ground Reaction Force): 발이 땅을 누를 때 땅이 몸을 밀어주는 힘이다. vertical, anterior-posterior(AP), medio-lateral(ML) 방향으로 나누어 볼 수 있다.
- COP(Center of Pressure): 지면반력이 발바닥의 어느 위치에 집중되는지를 나타내는 압력 중심 경로이다.
- EMG(Electromyography): 근육에서 발생하는 전기적 활동을 기록한 신호이다. 특정 근육이 언제, 얼마나 활성화되는지 확인할 수 있다.

## 4. 활용 데이터셋

### 4.1 메인 데이터셋: GaitRec

출처: https://www.nature.com/articles/s41597-020-0481-z

GaitRec은 근골격계 손상 환자와 건강 대조군의 보행 지면반력 데이터셋이다.

제공 규모:

- 근골격계 손상 환자: 2,084명
- 건강 대조군: 211명
- 양측 보행 trial: 75,732개

주요 라벨:

- Healthy control
- Hip
- Knee
- Ankle
- Calcaneus

라벨 의미:

이 라벨은 특정 근육이 약하다는 뜻이 아니라, 정형외과적 impairment가 주로 어느 부위에 있는지를 나타낸다. 예를 들어 Hip 라벨은 둔근 약화를 직접 뜻하지 않고, 골반/고관절/대퇴부 손상, 고관절 치환, coxarthrosis 등 고관절 주변 impairment를 포함한다.

제공 정보:

- Vertical GRF
- Anterior-posterior GRF
- Medio-lateral GRF
- AP 방향 COP
- ML 방향 COP
- left/right foot 데이터
- affected side
- 나이, 성별, 체중, 신장
- 신발 조건
- 보행 속도 등 metadata

### 4.2 보조 데이터셋: SIAT-LLMD

출처: https://www.nature.com/articles/s41597-023-02263-3

SIAT-LLMD는 건강한 40명의 하지 움직임 데이터를 제공하는 보조 데이터셋이다. GaitRec에는 EMG가 없기 때문에, EMG와 관절각/관절토크/GRF의 관계를 설명하는 참고 자료로 활용한다.

제공 정보:

- 대상자: 건강한 성인 40명
- 동작: 16개 하지 움직임
- sEMG
- 관절각
- 관절토크
- GRF
- active/rest label
- walking, upstairs, downstairs의 gait phase label

주의:

SIAT-LLMD는 건강인 데이터이므로 "정상 움직임 정답지"로 쓰면 안 된다. 이 데이터는 건강한 성인에서 관찰되는 하지 근활성 타이밍과 관절 움직임 관계를 참고하는 용도로 사용한다.

## 5. 핵심 로직 플로우

### Step 1. GaitRec에서 보행 기능 지표 추출

GaitRec의 left/right GRF, COP와 affected side 정보를 이용해 다음 지표를 계산한다.

- affected side와 unaffected side의 체중부하 차이
- vertical GRF peak 차이
- loading response 차이
- push-off 감소
- braking/propulsion force 비대칭
- COP path 길이와 이동 방향
- COP AP/ML 이동 범위
- stance phase에서 좌우 비대칭

### Step 2. impairment label과 보행 지표의 관계 분석

계산한 보행 지표가 Hip, Knee, Ankle, Calcaneus impairment group에서 어떻게 달라지는지 분석한다.

예시:

- Hip group: affected side의 체중부하 감소, 보행 속도 저하, push-off 변화가 나타나는지 확인
- Knee group: vertical loading, braking/propulsion 패턴, 좌우 GRF 비대칭 확인
- Ankle/Calcaneus group: COP 이동, forefoot push-off, AP force 패턴 확인

### Step 3. 모델은 "분류"보다 "설명"에 초점을 둔다

모델은 Healthy/Hip/Knee/Ankle/Calcaneus를 예측할 수 있지만, 프로젝트의 핵심은 높은 정확도 자체가 아니다. 어떤 지표가 특정 impairment label과 관련되는지 설명하고, 그 지표가 재활적으로 무엇을 의미할 수 있는지 해석하는 것이 중요하다.

가능한 설명 방식:

- feature importance
- SHAP
- class별 평균 GRF/COP curve 비교
- affected/unaffected side 차이 시각화
- confusion matrix로 어떤 impairment가 서로 비슷한 보행 패턴을 보이는지 확인

### Step 4. SIAT-LLMD로 EMG 해석을 보조한다

SIAT-LLMD는 GaitRec과 같은 대상자의 데이터가 아니므로 직접 결합해 하나의 진단 모델을 만들면 안 된다. 대신 다음처럼 보조 설명에 사용한다.

- gait phase별 하지 근육 활성 타이밍 확인
- hip/knee/ankle joint angle, torque와 EMG 변화의 관계 확인
- 특정 보행 지표가 어떤 근활성 또는 관절 운동과 관련될 수 있는지 생리학적 배경 설명

예시:

GaitRec에서 push-off 감소가 관찰될 경우, SIAT-LLMD의 보행 데이터에서 toe-off 전후 ankle plantarflexion torque와 gastrocnemius/soleus activation이 어떤 관계를 보이는지 참고할 수 있다.

## 6. 의학/재활 지식을 활용하는 위치

이 프로젝트에서 의학적 지식은 "진단 확정"이 아니라 "해석과 재활 초점 제안"에 사용한다.

활용 가능한 지식 예시:

- affected side의 GRF 감소는 통증 회피, 체중부하 회피, 근력 저하, 관절 불안정성의 후보 신호일 수 있다.
- push-off 감소는 발목 plantarflexor 기능, 족부/발목 통증, 보행 추진력 저하와 관련될 수 있다.
- COP path의 불안정성 또는 medial-lateral 이동 증가는 균형 조절, 발/발목 안정성, 하지 정렬 문제와 관련될 수 있다.
- 좌우 비대칭은 단순히 손상 부위뿐 아니라 보상 전략 또는 보호성 보행 전략을 반영할 수 있다.

## 7. 최종 결과물의 적절한 표현

강하게 말할 수 있는 결론:

- 이 보행은 특정 impairment group과 유사한 GRF/COP 패턴을 보인다.
- affected side에서 체중부하 회피 또는 추진력 감소가 관찰된다.
- Hip/Knee/Ankle/Calcaneus group별로 특징적인 보행 지표 차이가 있다.
- 재활 평가에서 우선 확인할 관절/기능 후보를 제안할 수 있다.

조심해야 하는 결론:

- 이 사람은 특정 근육이 약해서 아프다.
- 둔근 약화가 원인이다.
- 햄스트링 단축 때문에 보행 이상이 발생했다.
- 이 데이터만으로 정확한 재활 처방을 내릴 수 있다.

권장 최종 표현:

"본 프로젝트는 특정 근육의 약화나 통증 원인을 확정하지 않는다. 대신 GRF/COP 기반 보행 지표를 통해 근골격계 impairment와 관련된 기능적 이상 패턴을 찾고, 재활 평가에서 우선 확인할 관절 및 기능 후보를 제안한다."

## 8. 구현 범위 제안

### MVP

GaitRec processed 데이터만 사용한다.

사용 파일:

- GRF metadata
- processed vertical GRF left/right
- processed AP GRF left/right
- processed ML GRF left/right
- processed COP AP left/right
- processed COP ML left/right

구현:

- class별 평균 curve 시각화
- affected/unaffected side 비대칭 지표 계산
- 간단한 분류 모델
- feature importance 또는 SHAP
- 재활 초점 해석

### 확장

SIAT-LLMD를 일부 사용해 EMG와 관절각/관절토크 관계를 보조 분석한다.

구현:

- gait phase별 EMG 평균 패턴
- ankle/knee/hip torque와 관련 근육 EMG 비교
- GaitRec 결과 해석의 생리학적 배경으로 활용

## 9. 데이터 용량 및 학습 부담

### GaitRec 용량

공식 Figshare API 기준 GaitRec 전체 파일 합계는 약 3.94GB이다.

구성:

- RAW CSV 전체: 약 2.56GB
- processed CSV 전체: 약 1.39GB
- metadata: 약 0.6MB
- import code: 매우 작음

프로젝트에서는 RAW 전체를 받을 필요가 없다. 먼저 processed 데이터만 사용하면 약 1.39GB로 시작할 수 있다.

더 작게 시작하는 방법:

- metadata + vertical GRF left/right만 사용: 약 261MB
- metadata + vertical/AP/ML GRF left/right 사용: 약 826MB
- metadata + processed GRF/COP 전체 사용: 약 1.39GB

### SIAT-LLMD 용량

공식 Figshare API 기준 SIAT-LLMD는 압축 파일 하나로 약 7.8GB이다.

주의:

- 보조 데이터셋이므로 처음부터 전체 다운로드할 필요는 없다.
- 프로젝트 발표 수준에서는 논문과 제공 변수 설명을 바탕으로 EMG 해석 배경을 설명하고, 시간이 남으면 일부 샘플만 분석하는 방식이 현실적이다.

### 학습 부담

GaitRec은 이미지/영상 데이터가 아니라 CSV 기반 시계열 데이터이므로, 딥러닝 이미지 프로젝트보다 저장 공간과 GPU 부담이 작다.

현실적인 구현:

- feature 기반 모델: Logistic Regression, Random Forest, XGBoost, LightGBM 등
- 입력: raw time-series 전체가 아니라 peak, impulse, symmetry, COP range 등 요약 feature
- 학습 환경: 일반 노트북 CPU로도 가능
- GPU: 필수 아님

주의:

- CSV는 파일 크기보다 pandas 메모리 사용량이 커질 수 있다.
- 1.39GB processed CSV를 모두 한 번에 메모리에 올리면 8GB RAM 환경에서는 부담될 수 있다.
- chunk loading, 필요한 파일만 선택, feature 추출 후 parquet/csv로 저장하는 방식이 좋다.

## 10. 프로젝트의 핵심 가치

이 프로젝트의 장점은 병명만 맞히는 것이 아니라, 보행 기능 이상을 재활적으로 해석한다는 점이다.

핵심 메시지:

"AI가 정형외과적 진단명을 대신 내리는 것이 아니라, 보행 중 힘 분포와 압력 중심의 비대칭을 분석해 재활 평가에서 확인해야 할 기능적 문제 후보를 제시한다."

## 11. 발표에서 강조할 한계

- GaitRec에는 EMG가 없기 때문에 근육별 원인을 직접 진단할 수 없다.
- SIAT-LLMD는 건강인 데이터이므로 정상/비정상 판정의 기준으로 쓰면 안 된다.
- GaitRec의 Hip/Knee/Ankle/Calcaneus 라벨은 손상 또는 impairment 위치이지 특정 근육 라벨이 아니다.
- 본 프로젝트의 출력은 진단이나 처방이 아니라 재활 평가의 우선순위 후보이다.

