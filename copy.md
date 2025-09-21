아래는 **현업에 바로 갖다 쓰기 쉬운 수준**으로 다듬은 MLflow 4대 구성요소별 베이스라인입니다. 디렉터리 템플릿, 실행 명령어, 실무 팁까지 포함했습니다.

---

# 0) 폴더 구조(권장)

```
mlflow-baseline/
├─ MLproject
├─ conda.yaml
├─ src/
│  ├─ train.py
│  ├─ data.py
│  └─ utils.py
├─ scripts/
│  ├─ register_and_stage.py
│  └─ serve_local.sh
└─ mlruns/           # (로컬 Tracking store; 자동 생성)
```

> 로컬부터 시작해서 필요 시 Tracking URI만 바꿔 원격 서버로 확장하는 전략을 권장합니다.

---

# 1) **Tracking** 베이스라인 (수집/전처리/학습/메트릭/아티팩트 로깅)

`src/train.py`

```python
import os
import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from utils import seed_everything, log_dir


def run_experiment(C: float = 1.0, max_iter: int = 200, test_size: float = 0.2, random_state: int = 42):
    """단일 실행(전처리+학습+평가)을 하나의 Run으로 기록."""

    # 1) Tracking 설정 (로컬 파일 스토어)
    mlflow.set_tracking_uri("file:./mlruns")  # 필요 시 http://<host>:<port> 로 교체
    mlflow.set_experiment("binary_clf_baseline")

    seed_everything(42)

    with mlflow.start_run(run_name=f"train-{int(time.time())}"):
        # 2) 파라미터 로깅
        mlflow.log_params({
            "model": "logistic_regression",
            "C": C,
            "max_iter": max_iter,
            "test_size": test_size,
            "random_state": random_state,
        })

        # 3) 데이터 로드/스플릿(데모: 유방암 이진분류)
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 4) 파이프라인 구성(스케일러+로지스틱)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1))
        ])

        # 5) 학습
        pipe.fit(X_train, y_train)

        # 6) 예측 및 핵심 KPI 로깅
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics = {
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob)
        }
        mlflow.log_metrics(metrics)

        # 7) 모델/시그니처 로깅 (재현/서빙 대비)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X_test[:5],
            registered_model_name=None  # Registry는 아래 4)에서 별도 다룸
        )

        # 8) 부가 아티팩트(리포트/설정 파일 등) 로깅
        report_dir = Path("reports"); report_dir.mkdir(exist_ok=True)
        with open(report_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log_dir(report_dir.as_posix(), artifact_subdir="reports")

        # 9) 태그로 실행 맥락 남기기(코드 버전/환경 등)
        mlflow.set_tags({
            "project": "mlflow-baseline",
            "env": os.getenv("ENV", "local"),
        })

        print("Logged metrics:", metrics)


if __name__ == "__main__":
    # 하이퍼파라미터는 CLI/MLproject에서 주입 가능
    run_experiment()
```

`src/utils.py`

```python
import os
import random
import numpy as np
import torch
import mlflow
from pathlib import Path


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def log_dir(dir_path: str, artifact_subdir: str = None):
    p = Path(dir_path)
    if p.exists():
        mlflow.log_artifacts(p.as_posix(), artifact_path=artifact_subdir)
```

> **포인트**: 에폭/배치 단위가 아닌 **실행 종료 시점 요약** 위주로 로깅 → 오버헤드 최소화.

---

# 2) **Projects** 베이스라인 (재현 가능한 실행)

`MLproject`

```yaml
name: mlflow-baseline

conda_env: conda.yaml

entry_points:
  train:
    parameters:
      C: {type: float, default: 1.0}
      max_iter: {type: int, default: 200}
      test_size: {type: float, default: 0.2}
      random_state: {type: int, default: 42}
    command: "python -m src.train --C {C} --max_iter {max_iter} --test_size {test_size} --random_state {random_state}"
```

`conda.yaml`

```yaml
name: mlflow-baseline
channels: [defaults, conda-forge]
dependencies:
  - python=3.10
  - pip
  - scikit-learn
  - numpy
  - pandas
  - pip:
      - mlflow
      - torch==2.2.2; platform_system != 'Windows'  # 선택: seed용(torch 미설치 환경은 자동 패스)
```

**실행**

```bash
# (레포 루트)
mlflow run . -e train -P C=0.5 -P max_iter=500
```

> **포인트**: 동료는 이 레포만 받으면 **같은 환경/명령으로 재현** 가능.

---

# 3) **Models** 베이스라인 (저장/서빙)

### (A) 모델 저장은 위 `train.py`에서 이미 수행

* `mlflow.sklearn.log_model(..., artifact_path="model")` → Run 아래 `model/`로 저장

### (B) 로컬 서빙(개발/PoC용)

`scripts/serve_local.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$1"  # 예: 1234567890abcdef1234567890abcdef
PORT="${2:-5000}"

# runs:/<RUN_ID>/model 경로의 모델을 REST API로 서빙
mlflow models serve \
  -m "runs:/${RUN_ID}/model" \
  -p "${PORT}" \
  --no-conda

echo "Serve started on http://127.0.0.1:${PORT}"
```

**요청 예시**

```bash
curl -X POST http://127.0.0.1:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[-0.3, 1.2, ...], [0.5, -0.9, ...]]}'
```

> **포인트**: `mlflow models serve`는 단일 프로세스 REST 서버(Flask 기반). 실서비스 트래픽은 별도 프레임워크/컨테이너 오케스트레이션을 권장.

---

# 4) **Model Registry** 베이스라인 (버전/스테이지 관리)

`scripts/register_and_stage.py`

```python
import sys
import mlflow
from mlflow.tracking import MlflowClient

"""
사용법:
python scripts/register_and_stage.py <RUN_ID> <MODEL_NAME> <STAGE>
예:
python scripts/register_and_stage.py 1234abcd binary_clf_lr Production
"""

run_id = sys.argv[1]
model_name = sys.argv[2]
stage = sys.argv[3]  # None | Staging | Production | Archived

mlflow.set_tracking_uri("file:./mlruns")  # 필요 시 원격으로 교체
client = MlflowClient()

# 1) Run의 model 아티팩트 경로
model_uri = f"runs:/{run_id}/model"

# 2) Registry에 등록(없으면 생성)
registered = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Registered model name={model_name}, version={registered.version}")

# 3) Stage 전환
client.transition_model_version_stage(
    name=model_name,
    version=registered.version,
    stage=stage,
    archive_existing_versions=True  # 동일 Stage의 기존 버전은 Archived로 이동
)
print(f"Transitioned {model_name} v{registered.version} to {stage}")
```

**프로덕션 모델 로딩(서빙/배치 코드에서)**

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
model = mlflow.pyfunc.load_model("models:/binary_clf_lr/Production")
# 또는 특정 버전:
# model = mlflow.pyfunc.load_model("models:/binary_clf_lr/3")

# 예측
# import pandas as pd
# preds = model.predict(pd.DataFrame(...))
```

> **포인트**: Registry를 통해 **Staging ↔ Production** 승격이 쉬워지고, 소비 측(서빙/배치)은 **고정된 URI**로 최신 프로덕션을 참조할 수 있음.

---

## 실행 순서 요약

1. **Tracking/학습**

   ```bash
   python -m src.train                  # 또는 mlflow run . -e train
   mlflow ui --port 8080                # http://localhost:8080
   ```
2. **서빙(개발/PoC)**

   ```bash
   bash scripts/serve_local.sh <RUN_ID> 5000
   ```
3. **Registry 등록 및 프로모션**

   ```bash
   python scripts/register_and_stage.py <RUN_ID> binary_clf_lr Production
   ```
4. **소비 측에서 로딩**

   ```python
   mlflow.pyfunc.load_model("models:/binary_clf_lr/Production")
   ```

---

## 실무 팁(간단하지만 효과 큰 것들)

* **오버헤드 최소화**: 배치마다 로그 금지, 에폭/실행 요약만 기록. 이미지/체크포인트는 샘플/최종본만.
* **일관된 태깅**: `project`, `env`, `git_sha`, `data_hash` 등을 태그로 남겨 추적성↑.
* **Projects 채택**: 협업/재현성 향상. 로컬/CI/CD에서 동일 명령으로 실행.
* **Registry 중심 운영**: 소비 코드는 `models:/<name>/<Stage>`로 고정 → 릴리즈/롤백 깔끔.
* **원격 전환**: 초반엔 `file:./mlruns`, 이후 `http://mlflow.<corp>.com` 등으로 전환.

필요하면 위 템플릿을 **당신의 컴포넌트 구조**(DataCollector/FeatureExtractor/Trainer/Tuner/Explainer)로 정확히 끼워 맞춘 버전으로도 리팩토링해드릴게요.
