#!/usr/bin/env python3
import os
import re
import shlex
import subprocess
import threading
from pathlib import Path

import torch
from google.cloud import storage


WORKSPACE = Path("/workspace")
DATA_ROOT = WORKSPACE / "data"
RESULTS_ROOT = WORKSPACE / "results"
RESUME_DIR = WORKSPACE / "resume"
WEIGHTS_DIR = WORKSPACE / "weights"
SPECS_DIR = WORKSPACE / "specs"

CHECKPOINT_RE = re.compile(r"model_epoch_(\d+).*\.pth$")
SYNC_INTERVAL_SECONDS = 5

ACTION = os.environ.get("ACTION", "train").strip().lower()
EXPERIMENT = os.environ.get("EXPERIMENT", "").strip().lower()
DATA_ROOT_URI = os.environ.get("DATA_ROOT_URI", "").strip().rstrip("/")
REAL_N_RAW = os.environ.get("REAL_N", "").strip()
PRETRAINED_MODEL_URI = os.environ.get("PRETRAINED_MODEL_URI", "").strip()
CHECKPOINT_URI = os.environ.get("CHECKPOINT_URI", "").strip()

SMOKE_TEST = (
    os.environ.get("SMOKE_TEST", "false").strip().lower()
    in {"1", "true", "yes", "y"}
)

AIP_CHECKPOINT_DIR = os.environ.get("AIP_CHECKPOINT_DIR", "").strip()
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "").strip()


def banner(message):
    print(f"\n{message}\n", flush=True)


def parse_positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise SystemExit(
            f"{name} must be a positive integer, got: {value!r}"
        )

    if parsed <= 0:
        raise SystemExit(f"{name} must be > 0, got: {parsed}")

    return parsed


def resolve_configuration():
    if ACTION not in {"train", "evaluate"}:
        raise SystemExit("ACTION must be one of: train, evaluate")

    if EXPERIMENT not in {"synthetic", "sim2real", "real"}:
        raise SystemExit(
            "EXPERIMENT must be one of: synthetic, sim2real, real"
        )

    if not DATA_ROOT_URI:
        raise SystemExit(
            "Missing required environment variable: DATA_ROOT_URI"
        )

    if not DATA_ROOT_URI.startswith("gs://"):
        raise SystemExit(
            f"DATA_ROOT_URI must be a gs:// URI, got: {DATA_ROOT_URI}"
        )

    config = {
        "real_n": None,
        "spec": None,
        "data_uri": None,
        "data_dir": None,
        "results_dir": None,
        "train_json": None,
        "pretrained_model_local": None,
        "evaluation_checkpoint_local": None,
    }

    if EXPERIMENT == "synthetic":
        config.update(
            {
                "spec": SPECS_DIR / "rtdetr_synthetic.yaml",
                "data_uri": f"{DATA_ROOT_URI}/synth",
                "data_dir": DATA_ROOT / "synth",
                "results_dir": RESULTS_ROOT / ("synthetic" if ACTION == "train" else "synthetic_eval"),
            }
        )
        return config

    if not REAL_N_RAW:
        raise SystemExit(
            f"REAL_N is required for EXPERIMENT={EXPERIMENT}"
        )

    real_n = parse_positive_int(REAL_N_RAW, "REAL_N")
    data_dir = DATA_ROOT / "real"

    if real_n == 337:
        train_json = data_dir / "annotations" / "real_train_full.json"
    else:
        train_json = (
            data_dir
            / "annotations"
            / f"real_train_n{real_n}.json"
        )

    config.update(
        {
            "real_n": real_n,
            "spec": SPECS_DIR / f"rtdetr_{EXPERIMENT}.yaml",
            "data_uri": f"{DATA_ROOT_URI}/real",
            "data_dir": data_dir,
            "results_dir": RESULTS_ROOT / (
                f"{EXPERIMENT}_n{real_n}"
                if ACTION == "train"
                else f"{EXPERIMENT}_n{real_n}_eval"
            ),
            "train_json": train_json,
        }
    )

    if ACTION == "train" and EXPERIMENT == "sim2real":
        if not PRETRAINED_MODEL_URI:
            raise SystemExit(
                "PRETRAINED_MODEL_URI is required for EXPERIMENT=sim2real"
            )

        if not PRETRAINED_MODEL_URI.startswith("gs://"):
            raise SystemExit(
                "PRETRAINED_MODEL_URI must be a gs:// URI, got: "
                f"{PRETRAINED_MODEL_URI}"
            )

        config["pretrained_model_local"] = (
            WEIGHTS_DIR / "synthetic_pretrained.pth"
        )

    if ACTION == "evaluate":
        if not CHECKPOINT_URI:
            raise SystemExit(
                "CHECKPOINT_URI is required for ACTION=evaluate"
            )

        if not CHECKPOINT_URI.startswith("gs://"):
            raise SystemExit(
                "CHECKPOINT_URI must be a gs:// URI, got: "
                f"{CHECKPOINT_URI}"
            )

        config["evaluation_checkpoint_local"] = (
            WEIGHTS_DIR / "evaluation_checkpoint.pth"
        )

    return config


CONFIG = resolve_configuration()

SPEC = CONFIG["spec"]
DATA_URI = CONFIG["data_uri"]
DATA_DIR = CONFIG["data_dir"]
RESULTS_DIR = CONFIG["results_dir"]
TRAIN_JSON = CONFIG["train_json"]
REAL_N = CONFIG["real_n"]
PRETRAINED_MODEL_LOCAL = CONFIG["pretrained_model_local"]
EVALUATION_CHECKPOINT_LOCAL = CONFIG["evaluation_checkpoint_local"]


def require_environment():
    banner("ENVIRONMENT")

    values = {
        "ACTION": ACTION,
        "EXPERIMENT": EXPERIMENT,
        "REAL_N": REAL_N,
        "SMOKE_TEST": SMOKE_TEST if ACTION == "train" else None,
        "DATA_ROOT_URI": DATA_ROOT_URI,
        "DATA_URI": DATA_URI,
        "SPEC": SPEC,
        "RESULTS_DIR": RESULTS_DIR,
        "TRAIN_JSON": TRAIN_JSON,
        "PRETRAINED_MODEL_URI": (
            PRETRAINED_MODEL_URI
            if ACTION == "train" and EXPERIMENT == "sim2real"
            else None
        ),
        "CHECKPOINT_URI": (
            CHECKPOINT_URI if ACTION == "evaluate" else None
        ),
        "AIP_CHECKPOINT_DIR": (
            AIP_CHECKPOINT_DIR if ACTION == "train" else None
        ),
        "AIP_MODEL_DIR": AIP_MODEL_DIR,
        "CLOUD_ML_JOB_ID": os.environ.get("CLOUD_ML_JOB_ID"),
    }

    for key, value in values.items():
        print(f"{key:24s}: {value}", flush=True)

    missing = []

    if ACTION == "train" and not AIP_CHECKPOINT_DIR:
        missing.append("AIP_CHECKPOINT_DIR")

    if not AIP_MODEL_DIR:
        missing.append("AIP_MODEL_DIR")

    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    uris = [AIP_MODEL_DIR]
    if ACTION == "train":
        uris.append(AIP_CHECKPOINT_DIR)

    for uri in uris:
        if not uri.startswith("gs://"):
            raise SystemExit(f"Expected gs:// URI, got: {uri}")

    if not SPEC.exists():
        raise SystemExit(f"RT-DETR spec not found in container: {SPEC}")


def probe_gpu():
    banner("GPU PROBE")

    print(f"torch          : {torch.__version__}", flush=True)
    print(f"cuda runtime   : {torch.version.cuda}", flush=True)
    print(f"cuda available : {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: no GPU visible to the container")

    count = torch.cuda.device_count()
    print(f"device count   : {count}", flush=True)

    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)

        print(
            f"[{index}] {torch.cuda.get_device_name(index)} | "
            f"sm_{major}{minor} | "
            f"{props.total_memory / 1024**3:.1f} GiB",
            flush=True,
        )

    subprocess.run(["nvidia-smi"], check=True)


def parse_gs_uri(uri):
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")

    bucket_name, _, prefix = uri[5:].partition("/")

    if not bucket_name:
        raise ValueError(f"Missing bucket name: {uri}")

    return bucket_name, prefix.rstrip("/")


def download_gcs_prefix(uri, local_root):
    banner("DOWNLOAD DATASET")

    bucket_name, prefix = parse_gs_uri(uri)

    print(f"source : {uri}", flush=True)
    print(f"target : {local_root}", flush=True)

    client = storage.Client()
    query_prefix = f"{prefix}/" if prefix else ""

    blobs = list(
        client.list_blobs(
            bucket_name,
            prefix=query_prefix,
        )
    )

    files = [blob for blob in blobs if not blob.name.endswith("/")]

    if not files:
        raise SystemExit(f"No files found under {uri}")

    local_root.mkdir(parents=True, exist_ok=True)

    total_bytes = 0

    for index, blob in enumerate(files, start=1):
        relative = blob.name[len(query_prefix):]
        target = local_root / relative

        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))

        if blob.size:
            total_bytes += blob.size

        if index % 100 == 0 or index == len(files):
            print(
                f"downloaded {index}/{len(files)} files",
                flush=True,
            )

    print(
        "dataset download complete: "
        f"{len(files)} files, "
        f"{total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )


def download_gcs_file(uri, local_path):
    bucket_name, object_name = parse_gs_uri(uri)

    if not object_name:
        raise SystemExit(
            f"Expected a GCS object URI, got bucket/prefix only: {uri}"
        )

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    if not blob.exists(client):
        raise SystemExit(f"GCS object does not exist: {uri}")

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"source : {uri}", flush=True)
    print(f"target : {local_path}", flush=True)

    blob.download_to_filename(str(local_path))


def validate_dataset_layout():
    banner("DATASET SANITY CHECK")

    if EXPERIMENT == "synthetic":
        required = [
            DATA_DIR / "images/train",
            DATA_DIR / "images/val",
            DATA_DIR / "annotations/instances_train.json",
            DATA_DIR / "annotations/instances_val.json",
        ]

        for path in required:
            if not path.exists():
                raise SystemExit(
                    f"Required synthetic dataset path missing: {path}"
                )

            print(f"OK: {path}", flush=True)

        train_images = sum(
            1
            for path in (DATA_DIR / "images/train").iterdir()
            if path.is_file()
        )

        val_images = sum(
            1
            for path in (DATA_DIR / "images/val").iterdir()
            if path.is_file()
        )

        print(f"train images : {train_images}", flush=True)
        print(f"val images   : {val_images}", flush=True)

        if train_images == 0 or val_images == 0:
            raise SystemExit(
                "Synthetic train/val image directories are empty"
            )

        return

    required = [
        DATA_DIR / "images",
        TRAIN_JSON,
        DATA_DIR / "annotations/real_val.json",
        DATA_DIR / "annotations/real_test.json",
    ]

    for path in required:
        if not path.exists():
            raise SystemExit(
                f"Required real dataset path missing: {path}"
            )

        print(f"OK: {path}", flush=True)

    image_count = sum(
        1
        for path in (DATA_DIR / "images").iterdir()
        if path.is_file()
    )

    print(f"real images : {image_count}", flush=True)
    print(f"train JSON  : {TRAIN_JSON}", flush=True)

    if image_count == 0:
        raise SystemExit("Real image directory is empty")


def prepare_pretrained_model():
    if not (ACTION == "train" and EXPERIMENT == "sim2real"):
        return

    banner("DOWNLOAD SYNTHETIC PRETRAINED MODEL")

    download_gcs_file(
        PRETRAINED_MODEL_URI,
        PRETRAINED_MODEL_LOCAL,
    )

    print(
        f"pretrained model ready: {PRETRAINED_MODEL_LOCAL}",
        flush=True,
    )


def prepare_evaluation_checkpoint():
    if ACTION != "evaluate":
        return

    banner("DOWNLOAD EVALUATION CHECKPOINT")

    download_gcs_file(
        CHECKPOINT_URI,
        EVALUATION_CHECKPOINT_LOCAL,
    )

    print(
        f"evaluation checkpoint ready: {EVALUATION_CHECKPOINT_LOCAL}",
        flush=True,
    )


def checkpoint_epoch(name):
    match = CHECKPOINT_RE.search(Path(name).name)

    if not match:
        return -1

    return int(match.group(1))


def remote_checkpoints():
    bucket_name, prefix = parse_gs_uri(
        AIP_CHECKPOINT_DIR
    )

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    query_prefix = f"{prefix}/" if prefix else ""

    blobs = [
        blob
        for blob in client.list_blobs(
            bucket_name,
            prefix=query_prefix,
        )
        if CHECKPOINT_RE.search(Path(blob.name).name)
    ]

    return bucket, blobs


def download_latest_checkpoint():
    banner("CHECK FOR RESUME")

    bucket, blobs = remote_checkpoints()

    if not blobs:
        print(
            "No previous checkpoint found. Starting a new run.",
            flush=True,
        )
        return None

    newest = max(
        blobs,
        key=lambda blob: checkpoint_epoch(blob.name),
    )

    RESUME_DIR.mkdir(parents=True, exist_ok=True)

    local = RESUME_DIR / Path(newest.name).name

    print(
        f"found checkpoint : gs://{bucket.name}/{newest.name}",
        flush=True,
    )

    newest.download_to_filename(str(local))

    print(f"downloaded to    : {local}", flush=True)
    print(
        f"resuming from epoch {checkpoint_epoch(newest.name)}",
        flush=True,
    )

    return local


def local_checkpoints():
    if not RESULTS_DIR.exists():
        return []

    return [
        path
        for path in RESULTS_DIR.rglob("model_epoch_*.pth")
        if path.is_file()
    ]


def upload_checkpoint(path):
    bucket_name, prefix = parse_gs_uri(
        AIP_CHECKPOINT_DIR
    )

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    object_name = (
        f"{prefix}/{path.name}"
        if prefix
        else path.name
    )

    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(path))

    print(
        "checkpoint uploaded: "
        f"gs://{bucket_name}/{object_name}",
        flush=True,
    )


def checkpoint_watcher(stop_event):

    banner("CHECKPOINT WATCHER")

    uploaded = set()
    previous_state = {}

    while not stop_event.is_set():
        for checkpoint in local_checkpoints():
            key = str(checkpoint)
            stat = checkpoint.stat()

            current_state = (
                stat.st_size,
                stat.st_mtime_ns,
            )

            if key in uploaded:
                continue

            if previous_state.get(key) == current_state:
                upload_checkpoint(checkpoint)
                uploaded.add(key)
            else:
                previous_state[key] = current_state

        stop_event.wait(SYNC_INTERVAL_SECONDS)

    for checkpoint in local_checkpoints():
        key = str(checkpoint)

        if key not in uploaded:
            upload_checkpoint(checkpoint)
            uploaded.add(key)


def build_train_command(resume_checkpoint):
    command = [
        "rtdetr",
        "train",
        "-e",
        str(SPEC),
        f"results_dir={RESULTS_DIR}",
    ]

    if EXPERIMENT in {"sim2real", "real"}:
        command.append(
            "dataset.train_data_sources.0.json_file="
            f"{TRAIN_JSON}"
        )

    if (
        EXPERIMENT == "sim2real"
        and resume_checkpoint is None
    ):
        command.append(
            "train.pretrained_model_path="
            f"{PRETRAINED_MODEL_LOCAL}"
        )

    if SMOKE_TEST:
        command.extend(
            [
                "train.num_epochs=5",
                "train.checkpoint_interval=1",
                "train.validation_interval=1",
            ]
        )

    if resume_checkpoint:
        command.append(
            "train.resume_training_checkpoint_path="
            f"{resume_checkpoint}"
        )

    return command


def run_training(resume_checkpoint):
    banner("TAO RT-DETR TRAINING")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = build_train_command(resume_checkpoint)

    print("command:", flush=True)
    print(shlex.join(command), flush=True)

    stop_event = threading.Event()

    watcher = threading.Thread(
        target=checkpoint_watcher,
        args=(stop_event,),
        daemon=True,
    )

    watcher.start()

    process = subprocess.Popen(command)
    return_code = process.wait()

    stop_event.set()
    watcher.join()

    if return_code != 0:
        raise SystemExit(
            f"TAO training failed with exit code {return_code}"
        )

    print(
        "TAO training completed successfully.",
        flush=True,
    )


def build_evaluate_command():
    return [
        "rtdetr",
        "evaluate",
        "-e",
        str(SPEC),
        f"results_dir={RESULTS_DIR}",
        f"evaluate.checkpoint={EVALUATION_CHECKPOINT_LOCAL}",
    ]


def run_evaluation():
    banner("TAO RT-DETR EVALUATION")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prepare_evaluation_checkpoint()
    command = build_evaluate_command()

    print("command:", flush=True)
    print(shlex.join(command), flush=True)

    process = subprocess.Popen(command)
    return_code = process.wait()

    if return_code != 0:
        raise SystemExit(
            f"TAO evaluation failed with exit code {return_code}"
        )

    print(
        "TAO evaluation completed successfully.",
        flush=True,
    )


def latest_local_checkpoint():
    checkpoints = local_checkpoints()

    if not checkpoints:
        raise SystemExit(
            "Training completed but no TAO checkpoint was found."
        )

    return max(
        checkpoints,
        key=lambda path: checkpoint_epoch(path.name),
    )


def upload_final_model():

    banner("FINAL MODEL")

    checkpoint = latest_local_checkpoint()

    bucket_name, prefix = parse_gs_uri(
        AIP_MODEL_DIR
    )

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    object_name = (
        f"{prefix}/{checkpoint.name}"
        if prefix
        else checkpoint.name
    )

    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(checkpoint))

    print(
        f"final checkpoint : {checkpoint}",
        flush=True,
    )
    print(
        "uploaded model   : "
        f"gs://{bucket_name}/{object_name}",
        flush=True,
    )


def main():
    title = (
        "VERTEX AI - TAO RT-DETR - "
        f"{ACTION.upper()} - {EXPERIMENT.upper()}"
    )

    if REAL_N is not None:
        title += f" N={REAL_N}"

    if ACTION == "train" and SMOKE_TEST:
        title += " [SMOKE TEST]"

    banner(title)

    require_environment()
    probe_gpu()

    download_gcs_prefix(
        DATA_URI,
        DATA_DIR,
    )

    validate_dataset_layout()

    if ACTION == "train":
        prepare_pretrained_model()
        resume_checkpoint = download_latest_checkpoint()
        run_training(resume_checkpoint)
        upload_final_model()
    else:
        run_evaluation()

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
