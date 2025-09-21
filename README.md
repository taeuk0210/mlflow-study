# How to use MLflow

## 📌 MLflow main components  

### 1️⃣ MLflow Tracking  

실험(Experiment)과 실행(Run)을 기록하고 관리하는 기능으로 파라미터, 메트릭, 아티팩트(모델, 이미지, 로그 등), 소스 코드 버전을 자동/수동으로 저장하는 기능

> “어떤 파라미터로 돌렸더니 성능이 어땠는지”를 기록하고 비교.  

**핵심 키워드**  
- `mlflow.log_param`
- `mlflow.log_metric`
- `mlflow.log_artifact`
- UI (`mlflow ui --port 5000`)에서 실험 비교
- Autologging 지원 (`scikit-learn`, `LightGBM`, `PyTorch` 등)

### 2️⃣ MLflow Projects

재현 가능한 실행 환경을 정의하는 표준으로 코드와 환경(Conda, Docker 등)을 묶어서 프로젝트를 실행하면 이전과 동일한 결과가 나올 수 있도록 보장.

> 협업/배포 시, 같은 코드를 누구나 동일하게 실행 가능하게 만듦.

**핵심 키워드**
- MLproject 파일: 진입점(entry point), Conda 환경, 파라미터 정의
- `mlflow run <repository>` → 깃허브 레포 그대로 실행

### 3️⃣ MLflow Models

학습된 모델을 여러 “Flavor”(형식)로 저장하고 불러오는 표준 규격으로 예를 들면, scikit-learn 모델을 저장하면 → PyFunc(범용), sklearn flavor 둘 다 기록됨.

> 모델을 재사용·배포하기 쉽게 해줌.

**핵심 키워드**
 - `mlflow.<{>framework>.log_model`
 - Flavor: `sklearn`, `pytorch`, `tensorflow`, `xgboost`, `pyfunc` 등
 - CLI로 REST API 서빙 가능 → `mlflow models serve -m <model>`

### 4️⃣ MLflow Model Registry

모델 버전 관리와 라이프사이클(Stage)을 관리하는 기능으로 `Staging` → `Production` → `Archived` 로 상태 전환 가능.

> 팀 단위 협업에서 “지금 배포된 모델이 어떤 버전인지” 추적 가능.

**핵심 키워드**
 - 등록: `mlflow.register_model`
 - 단계(Stage): `None`, `Staging`, `Production`, `Archived`
 - UI에서 모델 비교/승격(Promotion) 가능

## 📌 MLflow baseline






## Q & A

**1. MLflow로 모델 서빙(배포)가 가능한가?**  

```
MLflow 자체적으로 serving 기능은 제공함, 기본적으로 REST API의 형태이고 `/invocations` 엔드포인트로 `JSON` 또는 `pandas.DataFrame` 형식을 POST 하면 예측 결과를 반환  
그러나, 보통은 MLflow는 Model registry, Artifact store로 사용하고 실제 서빙은 전용 서빙 프레임워크를 이용
```

**2. MLflow만으로 MLOps 파이프라인 구축이 가능한가?**  

```
1️⃣ MLflow 단독으로 할 수 있는 것  
- Experiment Tracking: 실험 기록(파라미터, 메트릭, 아티팩트, 코드 버전 등)  
- Model Packaging & Reproducibility  
- MLflow Projects (MLproject, conda.yaml, Docker) → 재현 실행  
- Model Management: MLflow Models (여러 프레임워크 모델 저장/로드/배포 표준화)  
- Model Registry (버전 관리, Stage 전환: Staging/Production/Archived)  
- Serving (PoC 수준)  
- mlflow models serve 명령으로 REST API 띄우기 (Flask 기반, 단일 프로세스)  

👉 여기까지는 MLflow만으로도 “작은 스케일의 ML lifecycle 관리”는 가능합니다.

2️⃣ MLflow 단독으로 부족한 부분

- 데이터 파이프라인/ETL: Airflow, Prefect, Dagster 같은 워크플로우 툴 필요
- CI/CD 자동화: GitHub Actions, GitLab CI, Jenkins, ArgoCD 같은 CI/CD 시스템 필요
- 대규모 모델 서빙: Kubernetes, Kubeflow Serving, Seldon, BentoML, Triton 같은 전용 서빙 프레임워크 필요
- 모니터링/알림: Prometheus, Grafana, ELK 스택, Sentry 같은 모니터링 시스템 필요
- Feature Store: Feast, Hopsworks 같은 피처 관리 솔루션 필요
- 데이터 버전 관리: DVC, LakeFS 같은 DataOps 도구 필요

👉 즉, MLflow는 Tracking & Registry 중심이고, Pipeline 오케스트레이션 / Production 서빙 / 모니터링까지는 직접 커버하지 못합니다.

3️⃣ 그래서 실무에서는 보통 이렇게 씀

- MLflow + Airflow: Airflow에서 데이터 가공/모델 학습 DAG 실행 → 각 태스크에서 MLflow로 로그 기록
- MLflow + CI/CD: 새로운 모델이 Registry에 Staging으로 등록되면 GitHub Actions/Jenkins가 자동 테스트 → 통과 시 Production으로 승격
- MLflow + Serving 프레임워크: Registry에서 Production 모델을 불러와 TorchServe/BentoML/Triton에서 운영
- MLflow + 모니터링: 모델 성능/드리프트 지표는 MLflow metric에 기록 + Prometheus/Grafana 연동
```