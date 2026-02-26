# LIBERO 기준 학습/운영 분리 파이프라인 설계

## 1. 목적
- `ricl_openpi_libero`의 LIBERO I/O 계약을 단일 기준으로 유지한다.
- 학습 경로는 기존 RICL 코드를 유지한다.
- 배포/추론 경로는 로컬 벡터DB + O/X 승인 + 팀 동기화를 사용한다.
- 데이터 소스는 `A 또는 B`를 선택해서 처리한다. (A+B 강제 통합 아님)

---

## 2. 기본 원칙
- Source A/B는 동시에 처리할 수 있지만, 한 실행 단위에서는 `A 또는 B`를 명시적으로 선택한다.
- Source가 무엇이든 전처리 출력 스키마는 동일해야 한다.
- 승인(O)된 데모만 운영용 벡터DB 및 동기화 대상으로 포함한다.
- 미승인 또는 실패(X) 데모는 추론/공유 경로에서 제외한다.

---

## 3. 정식 데이터 계약 (Canonical I/O)

### 3.1 전처리 출력 파일
- 파일: `processed_demo.npz`
- 필수 키:
`state`, `actions`, `base_image`, `wrist_image`, `base_image_embeddings`, `wrist_image_embeddings`, `prompt`
- 필수 shape:
`state: (T, 8)`, `actions: (T, 7)`, `base_image/wrist_image: (T, 224, 224, 3)`

### 3.2 추론 요청 입력
- `query_base_image`
- `query_wrist_image`
- `query_state`
- `query_prompt`

### 3.3 추론 출력
- `query_actions`

---

## 4. 소스별 전처리 (A 또는 B)

### 4.1 Source A (기존 LeRobot LIBERO 데이터셋)
- 입력:
`physical-intelligence/libero` (parquet episodes)
- 처리:
`preprocessing/process_libero_demos.py`
- 출력:
`preprocessing/libero_collected_demos_<tag>/<task>/<episode>/processed_demo.npz`
- 의미:
task = 상위 명령 단위, episode = 하위 데모 단위

### 4.2 Source B (실제 로봇 수집 데이터)
- 입력:
로봇 raw 로그(카메라/상태/액션)
- 처리:
`preprocessing/process_robot_demos_to_libero.py`
- 출력:
Source A와 동일한 구조의 `processed_demo.npz`

실행 예시:
```bash
cd preprocessing

# 단일 group 디렉터리 처리
uv run process_robot_demos_to_libero.py --dir runtime_demos/pending/robot_a_task_pick

# 여러 group 디렉터리 처리
uv run process_robot_demos_to_libero.py --dir_of_dirs runtime_demos/pending
```

입력 디렉터리 계약(episode 기준):
```
<episode_dir>/
  trajectory.h5 or traj.h5
  recordings/frames/varied_camera_1/*.png|jpg
  recordings/frames/hand_camera/*.png|jpg
  meta.json (optional: prompt)
```

---

## 5. 학습 경로 (유지)

### 5.1 단계
1. Source A(또는 학습용으로 준비된 B)를 `processed_demo.npz`로 준비
2. `preprocessing/retrieve_within_collected_demo_groups.py` 실행
3. `scripts/setup_norm_states_for_ricl.py --env=libero --embedding_type=base_image`
4. `scripts/train_pi0_fast_ricl.py pi0_fast_libero_ricl ...`

### 5.2 비고
- 학습 경로에서는 `indices_and_distances_base_image.npz`가 필요하다.
- 학습 경로는 기존 `ricl_openpi_libero` 코드 흐름을 바꾸지 않는다.

---

## 6. 운영/배포 경로 (신규 정책)

### 6.1 Inference Validation Loop
1. 입력:
`processed_demo.npz` + `serve_policy_ricl.py --policy.ricl_env=libero`
2. 처리:
추론 루프 수행, 로그/영상 저장
3. 사람 O/X 판단
4. 결과:
`O -> approved`, `X -> rejected`

### 6.2 로컬 벡터DB 반영
1. 반영 시점:
`O(승인)` 직후
2. 저장 단위:
timestep 임베딩(`base_image_embeddings`) + 메타(`prompt`, episode/task 식별자, npz 참조)
3. 검색 대상:
승인된 샘플만 포함

### 6.3 Sync Upload Stage
1. 입력:
`approved`에서 생성된 벡터DB delta 또는 승인된 `processed_demo.npz` 패키지
2. 처리:
암호화 후 push, 서버 버전 증가 관리
3. pull:
다른 노드는 증분 pull 후 로컬 벡터DB 갱신

### 6.4 Serving Runtime
1. query 입력 수신
2. query embedding 생성
3. 로컬 벡터DB에서 top-k retrieval
4. retrieved context + query를 policy 입력으로 구성
5. `query_actions` 반환

---

## 7. 디렉터리 제안 (운영 기준)
- `runtime_demos/pending/<source_tag>/<task>/<episode>/processed_demo.npz`
- `runtime_demos/approved/<source_tag>/<task>/<episode>/processed_demo.npz`
- `runtime_demos/rejected/<source_tag>/<task>/<episode>/processed_demo.npz`
- `runtime_logs/<date>/<task>/<episode>/...`
- `local_vector_db/<team_id>/...`

---

## 8. 기존 설계 폐기 항목
- DROID 전처리 스키마(`actions=8`, `top_image/right_image`)를 운영 기준에서 제외
- `traj.h5 -> process_collected_demos.py`를 LIBERO 운영 파이프라인의 기본 경로로 사용하지 않음
- 승인 이전 데이터 자동 업로드 금지

---

## 9. 운영 정책 체크리스트
- O/X 판정 이력 저장 (`who`, `when`, `reason`)
- 승인 전/후 데이터 분리 저장
- sync 전 암호화 및 버전 기록 필수
- pull 실패 시 재시도와 멱등성 보장
- 로컬 벡터DB와 파일 저장소 간 참조 무결성 점검

---

## 10. 구현 우선순위
1. Source A/B 공통 `processed_demo.npz` validator (완료: `preprocessing/validate_processed_demo.py`)
2. O/X 승인 루프 스크립트 (완료: `preprocessing/review_and_route_demos.py`)
3. 승인 데이터 -> 로컬 벡터DB upsert adapter (완료: `preprocessing/build_local_vector_db.py`)
4. sync push/pull adapter
5. `RiclLiberoPolicy.retrieve()`를 파일기반 인덱스에서 벡터DB adapter 호출로 점진 이행

---

## 11. 신규 스크립트 실행 예시

1) `processed_demo.npz` 계약 검증
```bash
cd preprocessing
uv run validate_processed_demo.py \
  --root runtime_demos/pending \
  --report-json runtime_demos/validation_report.json
```

2) 수동 O/X 승인 라우팅
```bash
cd preprocessing
uv run review_and_route_demos.py \
  --pending-root runtime_demos/pending \
  --approved-root runtime_demos/approved \
  --rejected-root runtime_demos/rejected \
  --log-path runtime_demos/review_log.jsonl
```

3) approved -> 로컬 벡터DB 산출
```bash
cd preprocessing
uv run build_local_vector_db.py \
  --approved-root runtime_demos/approved \
  --output-root local_vector_db \
  --team-id team_a \
  --overwrite
```

---

## 12. 증분 동기화 CLI (`data`)

요구한 형태대로 CLI 엔트리포인트를 추가했다.

기본 명령:
```bash
uv run data upload <team-name-or-id> <task-name> <data-id>
```

인자 의미:
- `<team-name-or-id>`: 팀 저장소 식별자
- `<task-name>`: 태스크 이름
- `<data-id>`: 업로드할 데모(episode) 식별자

실사용 예시:
```bash
# 1) 업로드 payload 확인만 (전송 안 함)
uv run data upload team_a pick_up_book demo_11 --dry-run \
  --dry-run-output preprocessing/runtime_demos/dryrun_payloads

# 2) 실제 증분 업로드 (동일 hash면 자동 skip)
uv run data upload team_a pick_up_book demo_11 \
  --base-url http://localhost:8000 \
  --token-file ~/.serve_token

# 3) 팀 동기화 pull
uv run data pull team_a \
  --base-url http://localhost:8000 \
  --token-file ~/.serve_token \
  --materialize \
  --materialize-approved-root preprocessing/runtime_demos/approved_synced

# 4) 업로드 상태 확인
uv run data status team_a

# 5) pull된 chunk를 나중에 수동 복원
uv run data materialize team_a \
  --approved-root preprocessing/runtime_demos/approved_synced
```

증분 기준:
- `preprocessing/.sync_state/<team>/upload_manifest.json`에 업로드된 파일 hash를 기록한다.
- 같은 `<task, data_id>`가 같은 hash면 다음 업로드는 자동으로 skip된다.
- `--force` 옵션으로 강제 재업로드할 수 있다.

---

## 13. Sync Pull 재구성 설계 (npz vs 벡터DB)

핵심 원칙:
- 벡터DB는 `processed_demo.npz` 원본 저장소가 아니다.
- 벡터DB에는 검색용 벡터 + 메타만 저장한다.
- 원본 데모(`processed_demo.npz`)는 파일/Blob 계층에서 보관하고, pull 후 복원한다.

권장 데이터 흐름:
1. 업로드 측:
   - 승인된 `processed_demo.npz`를 chunk로 push
   - 파일명에 `<task>__<data_id>__processed_demo.npz` 규칙 포함
2. 서버:
   - chunk 단위 암호화 저장
3. pull 측:
   - 변경 chunk를 `preprocessing/.synced_chunks/<team>/<documentId>/`로 저장
   - 문서 레지스트리(`preprocessing/.sync_state/<team>/pull_docs.json`) 갱신
4. materialize 단계:
   - chunk를 순서대로 붙여 `processed_demo.npz` 재구성
   - `preprocessing/runtime_demos/approved_synced/<task>/<data_id>/processed_demo.npz` 생성
5. 인덱싱 단계:
   - 복원된 npz를 입력으로 로컬 벡터DB(embedding index) 갱신

왜 이 구조가 필요한가:
- pull 응답은 chunk 단위라 바로 policy retrieval 컨텍스트를 만들기 어렵다.
- RICL 추론 컨텍스트(`state/base_image/wrist_image/actions/prompt`)는 결국 `processed_demo.npz` 단위 참조가 가장 단순하고 안전하다.
- 이후 벡터DB 어댑터 전환 시에도, 원본 npz는 디버깅/재색인/감사 추적의 기준본이 된다.

실행 순서 예시 (pull 이후):
```bash
cd preprocessing

# 1) pull + npz 복원
uv run data pull team_a \
  --base-url http://localhost:8000 \
  --token-file ~/.serve_token \
  --materialize \
  --materialize-approved-root runtime_demos/approved_synced

# 2) 로컬 벡터DB 인덱스 갱신
uv run build_local_vector_db.py \
  --approved-root runtime_demos/approved_synced \
  --output-root local_vector_db \
  --team-id team_a \
  --overwrite
```

`RiclLiberoPolicy` 전환:
- `--policy.demos_dir=preprocessing/local_vector_db/<team_id>`를 넘기면 local vector DB 모드로 retrieval 동작
- `--policy.demos_dir=preprocessing/runtime_demos/approved...`를 넘기면 기존 demos_dir 모드로 동작

---

## 14. 실행 계약 예시 (구현 완료 기준)

### 14.1 Source B 전처리 계약
- 입력:
  - `<episode_dir>/trajectory.h5` or `<episode_dir>/traj.h5`
  - `<episode_dir>/recordings/frames/varied_camera_1/*`
  - `<episode_dir>/recordings/frames/hand_camera/*`
- 출력:
  - `<episode_dir>/processed_demo.npz`
  - `<episode_dir>/episode_meta.json`

```bash
cd preprocessing
uv run process_robot_demos_to_libero.py --dir runtime_demos/pending/sample_one
```

### 14.2 검증 계약
- 입력: `runtime_demos/pending/**/processed_demo.npz`
- 출력: `validation_report.json` + 종료코드(실패 시 2)

```bash
cd preprocessing
uv run validate_processed_demo.py \
  --root runtime_demos/pending \
  --report-json runtime_demos/validation_report.json
```

### 14.3 추론결과 기반 O/X 라우팅 계약
- 입력:
  - `pending` 데모
  - 추론 결과 폴더(`--results-root`) 또는 에피소드별 추론 명령(`--run-inference-cmd`)
- 처리:
  - 각 episode에 대해 `results_dir`와 artifact 목록을 표시
  - 사람 O/X 입력
- 출력:
  - `approved` 또는 `rejected`로 이동
  - `review_log.jsonl` 기록(결과 경로/아티팩트 수 포함)

```bash
cd preprocessing
uv run review_and_route_demos.py \
  --pending-root runtime_demos/pending \
  --approved-root runtime_demos/approved \
  --rejected-root runtime_demos/rejected \
  --results-root runtime_logs/inference \
  --require-results \
  --log-path runtime_demos/review_log.jsonl
```

`--run-inference-cmd` 템플릿 변수:
- `{episode_dir}`: 현재 검토 episode 경로
- `{rel_path}`: pending 기준 상대 경로
- `{result_dir}`: 결과 저장 대상 경로

예시:
```bash
cd preprocessing
uv run review_and_route_demos.py \
  --pending-root runtime_demos/pending \
  --approved-root runtime_demos/approved \
  --rejected-root runtime_demos/rejected \
  --results-root runtime_logs/inference \
  --run-inference-cmd "uv run python scripts/run_episode_eval.py --episode {episode_dir} --out {result_dir}" \
  --require-results
```

### 14.4 approved -> local vector DB 계약
- 입력: `runtime_demos/approved/**/processed_demo.npz`
- 출력: `local_vector_db/<team_id>/{summary.json,episodes.json,vectors.npz}`

```bash
cd preprocessing
uv run build_local_vector_db.py \
  --approved-root runtime_demos/approved \
  --output-root local_vector_db \
  --team-id team_a \
  --overwrite
```

### 14.5 serve runtime 계약
- 입력: `--policy.demos_dir=preprocessing/local_vector_db/<team_id>`
- 출력: 정책 서버 `query_actions`

```bash
cd ..
uv run scripts/serve_policy_ricl.py policy:checkpoint \
  --policy.config=pi0_fast_libero_ricl \
  --policy.dir=checkpoints/pi0_fast_libero_ricl/<EXP>/<STEP> \
  --policy.demos_dir=preprocessing/local_vector_db/team_a \
  --policy.ricl_env=libero \
  --port=8000
```
