# IMU-added Experiment Data

| 파일 | 내용 |
|---|---|
| `raw_sessions_archive-1.zip` | 센서에서 직접 수집한 원본 데이터 1부 |
| `raw_sessions_archive-2.zip` | 센서에서 직접 수집한 원본 데이터 2부 |
| `derived_dataset_archive.zip` | 원본을 trial 단위로 정리한 데이터셋 |
| `ml_preparation_v1.zip` | 정제하고 학습/검증용으로 분할한 ML 입력 데이터 |

```text
원본 수집 → trial 데이터셋 → ML 입력 데이터
```

학습된 모델과 평가 결과는
`../../05_ML/model_results/`에 보관한다.

## 원본 세션 ZIP 분할 안내

원래의 `raw_sessions_archive.zip`은 3.27 GB로 GitHub Free/Pro의 Git LFS
단일 파일 제한을 넘기 때문에 두 개의 독립 ZIP으로 나눴다. 두 ZIP을
같은 대상 폴더에 차례대로 압축 해제하면 원본의 전체 504개 파일이
복원된다. 두 ZIP 사이에 중복된 파일은 없다.

검증 결과:

| 파일 | 크기(byte) | SHA-256 |
|---|---:|---|
| `raw_sessions_archive-1.zip` | 1,636,954,952 | `BA36D4557B7A0C91E5003D5039BB47A6044D1BB78C30E566CA490131691EFBC8` |
| `raw_sessions_archive-2.zip` | 1,636,928,679 | `A94A8D6EBE1E009A9BE64B1E3E897C01B34F74530709DEADF7FC2B04C00E32C9` |

로컬에 보존된 분할 전 원본의 SHA-256은
`09846226C278970386F4A8CA21EA6EE3B5355156B46F70FABA95CC3B95B4FB71`이다.
원본 ZIP 자체의 바이트 배열이 아니라, 압축 해제된 파일 집합을 기준으로
두 분할 ZIP의 합집합과 원본이 일치함을 검증했다.
