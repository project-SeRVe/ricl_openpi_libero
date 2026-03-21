# LIBERO RICL 세팅 방법

## 초기 세팅

1. 다음 명령어를 실행한다. (RICL installation 명령어)

```shell
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install tensorflow-datasets tensorflow-cpu autofaiss google-genai openai
```

2. git submodule init, git submodule update를 실행한다. (libero를 third party로추가)
3. third_party/libero/libero에 빈 __init__.py 파일을 생성한다.
4. 다음 명령어로 xformers와 CUDA 12.8을 위한 pytorch를 설치한다. (만약 CUDA 버전이 다른 경우에는 CUDA 12.8 대신 적절하게 버전을 맞춰 줘야 한다.)

```shell
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
```

하지만 GPU를 P4를 사용하는 경우(옛날 GPU라 호환이 안됨), xformers는 사용하지 말고, 다음 명령어로 CUDA 11.8을 위한 pytorch를 설치해 사용한다. elice를 사용하는 경우 CUDA 12.4가 default이므로 이 경우에도 CUDA 11.8을 위한 pytorch를 쓰자.

```shell
# torch, xformers 삭제
uv pip uninstall torch torchvision torchaudio xformers

# CUDA 11.8 pytorch 설치
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

만약 세팅을 다 해도 libero를 import하는 코드에 module 인식이 안 된다는 경고가 뜨면 (vscode 문제) .vscode/settings.json에 다음 항목을 추가한다.

```json
{
    "python.analysis.extraPaths": [
            "~/jhun/capstone/ricl_openpi_edited/third_party/libero"
        ]
}
```

참고로 third_party/libero와 examples/libero에도 세팅 방법이 나와 있지만, RICL 세팅과 함께 적용할 경우 버전 충돌이 나므로 위의 방법만 따라하자.

## GCP 인스턴스를 껐다가 킨 경우의 세팅

(elice 등 다른 platform을 사용하는 경우 무시하자)

1. GCP 인스턴스를 껐다가 연결할 때 외부 IP 주소가 변경되므로, 변경된 값으로 ssh config 파일을 수정해야 한다.
2. 외부 디스크를 다시 mount해줘야 한다.

```bash
# 디스크 이름 확인
lsblk

# mount (첫 번째 인자에 디스크 이름 지정)
sudo mount /dev/sdb /mnt/disks/sdb
```

# LIBERO 구현 이후 명령어

다음과 같은 명령어와 순서로 데이터 전처리, 정규화 통계 생성, training, serving, evaluation을 수행할 수 있음.

```shell                            
1. 데이터 전처리

cd preprocessing

# 전체 LIBERO 데이터 처리 (LeRobot HF 데이터셋 → processed_demo.npz)
# 압축된 버전으로 .npz 파일 저장
uv run --no-sync process_libero_demos.py --output_dir=libero_collected_demos --compressed

# 학습용으로 사용할 task 개수만큼 폴더 이동 (source: libero_collected_demos -> target: libero_collected_demos_training)
# 전체 task는 40개인 점을 고려해서 숫자를 지정하자
uv run --no-sync select_libero_train_tasks.py --num_tasks=32
# (선택) task 선택 순서를 랜덤으로 섞고 싶으면 대신 다음 명령어 실행
uv run --no-sync select_libero_train_tasks.py --num_tasks=32 --shuffle --seed=31

# 위 이동을 되돌리고 싶을 때 (source: libero_collected_demos_training -> target: libero_collected_demos)
uv run --no-sync unselect_libero_train_tasks.py --num_tasks=32


# 학습용 retrieval 인덱스 생성 (indices_and_distances_base_image.npz, libero_collected_demos_training 기준)
uv run --no-sync retrieve_within_collected_demo_groups.py \
  --folder_name=libero_collected_demos_training \
  --embedding_type=base_image

---
2. 정규화 통계 생성 (프로젝트 루트에서)

cd ..  # 프로젝트 루트로 복귀

uv run --no-sync scripts/setup_norm_states_for_ricl.py --env=libero --embedding_type=base_image

출력 파일:
- assets/norm_stats_simple_libero.json
- assets/max_distance_libero.json  ← 서빙 시 RiclLiberoPolicy가 직접 참조
- assets/libero/norm_stats.json    ← 학습 시 사용

---
3. Training

# wandb 버전 수정
uv pip install "wandb>=0.22.3"

# cannot open the shared object file 오류가 나는 경우 다음 의존성 설치
sudo apt-get update
sudo apt-get install -y libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1

# 학습 전에 jax 환경변수 지정하기
printf '\nexport XLA_PYTHON_CLIENT_PREALLOCATE=false\nexport XLA_PYTHON_CLIENT_ALLOCATOR=platform\n' >> ~/.bashrc
source ~/.bashrc
echo $XLA_PYTHON_CLIENT_PREALLOCATE
echo $XLA_PYTHON_CLIENT_ALLOCATOR

# 학습(priming) 스크립트
uv run --no-sync scripts/train_pi0_fast_ricl.py pi0_fast_libero_ricl \
  --exp-name=priming \
  --overwrite

# fine-tuning 스크립트 (우리 프로젝트에선 x)
uv run --no-sync scripts/train_pi0_fast_ricl.py pi0_fast_libero_ricl___finetune_on_new_task \
  --exp-name=finetuning \
  --overwrite

---
4. Serving/Evaluation

LIBERO RICL의 serving/evaluation은 두 환경을 분리해서 실행해야 한다.

- policy server: 프로젝트 루트 `.venv`
- LIBERO evaluation: `examples/libero/.venv`

이는 `serve_policy_ricl.py`는 위에서 세팅한 RICL 환경을 요구하고, `examples/libero/main_ricl.py`는 LIBERO simulator 의존성을 요구하기 때문이다. 하나의 Python 환경에서 같이 돌리면 버전 충돌이 난다. 다음 instruction에 따라서 libero 환경을 세팅하고, 그 아래의 셸 스크립트로 이 두 환경을 나눠서 자동으로 실행한다.

1) LIBERO evaluation용 환경 생성

```shell
cd /home/elicer/capstone/ricl_openpi_libero

uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
# 참고로, third_party와 example의 libero robosuite 버전이 안 맞아서, example의 것을 third_party에 맞춰줬다. (1.4.1->1.4.0)

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl

# 이 부분은 위의 세팅으로도 안되면 사용을 검토하자
# export LIBERO_CONFIG_PATH=$PWD/.libero
# mkdir -p $LIBERO_CONFIG_PATH
# cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
# benchmark_root: $PWD/third_party/libero/libero/libero
# bddl_files: $PWD/third_party/libero/libero/libero/bddl_files
# init_states: $PWD/third_party/libero/libero/libero/init_files
# datasets: $PWD/third_party/libero/libero/datasets
# assets: $PWD/third_party/libero/libero/libero/assets
# EOF
```

3) evaluation 환경 검증

```shell
python -c "from libero.libero import benchmark, get_libero_path; print(get_libero_path('bddl_files'))"
python -c "import robosuite, mujoco, bddl, gym, robomimic, hydra; print('libero env ok')"
```

4) 스크립트로 serving/eval 수행

- `scripts/run_libero_ricl_servers.sh`는 policy server를 루트 `.venv`의 Python으로 실행하고, LIBERO eval은 `examples/libero/.venv`의 Python으로 실행하는 serving/eval 통합 실행 스크립트이다.
- `--task-name`을 생략하면 `preprocessing/libero_collected_demos` 아래의 모든 task를 task별로 개별 평가한다.
- `--task-name=<task_name>` 또는 기존 별칭인 `--task=<task_name>`을 주면 해당 task만 평가한다.
- 배치 러너는 task마다 policy server를 다시 띄우므로, 각 task 평가 시 해당 task의 retrieval demo만 사용한다.
- 필요하면 `--server-python`, `--eval-python`으로 각 환경의 Python 경로를 직접 지정할 수 있다.
- 이 스크립트를 쓰는 경우 `serve_policy_ricl.py`와 `main_ricl.py`를 따로 실행할 필요가 없다.

모든 task를 개별 평가하는 예시:

```shell
./scripts/run_libero_ricl_servers.sh \
  --demos-root=preprocessing/libero_collected_demos \
  --checkpoint-dir=checkpoints/pi0_fast_libero_ricl/priming/3600 \
  --num-trials-per-task=10 \
  --video-out-root=examples/libero/data/libero_ricl/batch_eval
```

특정 task만 평가하는 예시:

```shell
./scripts/run_libero_ricl_servers.sh \
  --demos-root=preprocessing/libero_collected_demos \
  --checkpoint-dir=checkpoints/pi0_fast_libero_ricl/priming/5700 \
  --num-trials-per-task=10 \
  --video-out-root=examples/libero/data/libero_ricl/batch_eval2 \
  --task-name=open_the_middle_drawer_of_the_cabinet
```
---

# RICL: Re-training (a VLA) for In-Context Learning
A RICL version of the openpi repository focused on RICL-Pi0-FAST-DROID.

[Website](https://ricl-vla.github.io/) | [Arxiv](https://arxiv.org/abs/2508.02062)

## Installation
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install tensorflow-datasets tensorflow-cpu autofaiss google-genai openai
```

## Quickstart
Collect retrieval data for a new task as detailed [under Collecting data > retrieval data for testing the VLA in a new task](#retrieval-data-for-testing-the-vla-in-a-new-task).

Then download our checkpoint and serve it with the above retrieval data as detailed [under Serving RICL-Pi0-FAST-DROID in a new task > Serve the downloaded checkpoint](#serve-the-downloaded-checkpoint)

## Collecting data
Collect demos on your franka droid robot. The demos must be setup in the following directory structures for priming (at training time) and retrieval (at testing time). Please use the [original droid code](https://droid-dataset.github.io/droid/) to collect the demos.

### Priming data for re-training a VLA for in-context learning
```bash
# The following example structure is expected for the collected demos:
# preprocessing/
# ├── collected_demos_training/
# │   ├── {YYYY-MM-DD}_{task_1_prompt}/
# │   │   ├── demo_0
# |   │   ├── demo_1
# |   │   ├── ...
# │   │── {YYYY-MM-DD}_{task_2_prompt}/
# │   │   ├── demo_0
# |   │   ├── demo_1
# |   │   ├── ...
```
Please note that the folder containing many demos for a task must have the aboev specified name format. Ensure that the task prompt uses underscores to separate words. The folders inside each task folder can be named anything. But note that each folder inside a task folder, which has all the information for one collected demo, must atleast contain the following:
```bash
# │   │   ├── demo_0
# │   │   │   ├── traj.h5
# │   │   │   ├── recordings/
# |   │   │   │   ├── frames/
# |   │   │   │   │   ├── hand_camera/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
# |   │   │   │   │   ├── varied_camera_1/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
# |   │   │   │   │   ├── varied_camera_2/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
```
where the number of jpg files (in each camera folder) is equal to the number of timesteps where the controller is active. You need to extract these from the svo files created by the droid code. The traj.h5 file is the one saved by the original droid code containing the proprioceptive data and actions. These are the only files we use. We ignore everything else in the demo folder.

### Retrieval data for testing the VLA in a new task
Similarly, for retrieval data in a new task at test time, we expect the following structure:
```bash
# preprocessing/
# ├── collected_demos/
# │   ├── {YYYY-MM-DD}_{new_task_prompt}/
# │   │   ├── demo_0
# |   │   ├── demo_1
# |   │   ├── ...
```

## Downloading our datasets
Priming (training) data: `git clone https://huggingface.co/datasets/ricl-vla/collected_demos_training ./preprocessing/collected_demos_training`

Retrieval (testing) data in many new tasks: `git clone https://huggingface.co/datasets/ricl-vla/collected_demos ./preprocessing/collected_demos`

Both of the above can also be found at [this huggingface link](https://huggingface.co/ricl-vla).

## Preprocessing [SKIP this step if you downloaded the above datasets from HF]
* First cd into the folder
```bash
cd preprocessing
```

* Process the priming demos for re-training a VLA for in-context learning
```bash
python process_collected_demos.py --dir_of_dirs=collected_demos_training
python retrieve_within_collected_demo_groups.py
```

* Process the retrieval demos for testing the VLA in a new task
```bash
python process_collected_demos.py --dir_of_dirs=collected_demos
```

## Re-training for in-context learning (RICL)
* Compute norm stats after processing the priming demos as follows **[SKIP this step if you downloaded the above datasets from HF]**:
```bash
python scripts/setup_norm_states_for_ricl.py
```

* Create RICL-Pi0-FAST-DROID
```bash
python scripts/train_pi0_fast_ricl.py pi0_fast_droid_ricl --exp-name={YOUR_EXPERIMENT_NAME_HERE} --overwrite
```

## Serving RICL-Pi0-FAST-DROID in a new task
### Serve RICL-Pi0-FAST-DROID's checkpoint after three epochs of training 
The step corresponding to this for our dataset is 5400.
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_droid_ricl --policy.dir=checkpoints/pi0_fast_droid_ricl/{YOUR_EXPERIMENT_NAME_HERE}/5400 --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{new_task_prompt}
```

### Serve the downloaded checkpoint
Download our checkpoint: `git clone https://huggingface.co/ricl-vla/pi0_fast_droid_ricl_checkpoint`

You can serve it as follows:
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_droid_ricl --policy.dir=pi0_fast_droid_ricl_checkpoint --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{new_task_prompt}
```

Our checkpoint can also be found at [this huggingface link](https://huggingface.co/ricl-vla).

### On the laptop connected to the franka droid robot
You also have to run the client in `examples/droid/main_ricl.py` on the laptop connexted to the franka droid robot following the original repository's instructions.

## Finetuning like you pretrain to create RICL-Pi0-FAST-DROID-Finetuned
* This requeires a bit more preprocessing of the retrieval demos in the new task
```bash
python retrieve_within_collected_demo_groups.py --folder_name=collected_demos
```

* Create RICL-Pi0-FAST-DROID-Finetuned
```bash
python scripts/train_pi0_fast_ricl.py pi0_fast_droid_ricl___finetune_on_new_task --exp-name={YOUR_EXPERIMENT_NAME_HERE} --overwrite
```
Please make sure the config of `pi0_fast_droid_ricl___finetune_on_new_task` in `src/training/config.py` points to the correct retrieval demos folder that you would like to finetune on.

* Serve RICL-Pi0-FAST-DROID-Finetuned
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_droid_ricl___finetune_on_new_task --policy.dir=checkpoints/pi0_fast_droid_ricl___finetune_on_new_task/{YOUR_EXPERIMENT_NAME_HERE}/999 --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{new_task_prompt}
```

Run the client on the laptop connected to the franka droid robot to test the finetuned policy.

## Credits
This repository is based on the [openpi](https://github.com/openai/openpi) repository, without which this work would not have been possible.
