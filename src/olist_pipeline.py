"""Заготовки решения: смотрите prototype.ipynb и ASSIGNMENT.md.

Адреса берите из DATABASE_URL, MLFLOW_TRACKING_URI, OLIST_WORKDIR.
Через XCom передавайте только пути, идентификаторы и короткие метаданные,
включая SHA256 снимка, но не таблицы, модели или содержимое файлов.
"""

import csv
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ("price", "freight", "weight", "n_items", "installments", "est_days")
TEXT_COLUMNS = (
    "order_id",
    "order_purchase_timestamp",
    "customer_state",
    "seller_state",
    "category",
    "payment_type",
)
REQUIRED_COLUMNS = {*TEXT_COLUMNS, *NUMERIC_COLUMNS, "late"}


def extract(run_key):
    """Прочитать public.orders, сохранить обычный UTF-8 CSV; вернуть абсолютный путь строкой."""
    import psycopg

    url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    workdir = Path(os.environ.get("OLIST_WORKDIR", "/opt/olist/work")).resolve()
    run_directory = hashlib.sha256(run_key.encode("utf-8")).hexdigest()
    snapshot = workdir / "runs" / run_directory / "snapshot.csv"
    snapshot.parent.mkdir(parents=True, exist_ok=True)

    query = """
        COPY (SELECT * FROM public.orders ORDER BY order_id COLLATE "C")
        TO STDOUT WITH (FORMAT CSV, HEADER true, ENCODING 'UTF8')
    """
    with TemporaryDirectory(dir=snapshot.parent) as directory:
        staged = Path(directory) / snapshot.name
        with psycopg.connect(url) as connection, connection.cursor() as cursor:
            with cursor.copy(query) as copy, staged.open("wb") as stream:
                for block in copy:
                    stream.write(block)
        staged.replace(snapshot)
    return str(snapshot)


def _read_csv(source):
    frame = pd.read_csv(source, encoding="utf-8", dtype=dict.fromkeys(REQUIRED_COLUMNS, "string"))
    for column in (*NUMERIC_COLUMNS, "late"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    return frame


def validate(snapshot):
    """Проверить файл snapshot до обучения; вернуть путь или вызвать исключение.

    Только чтение CSV: без записей, отчётов, mkdir/chdir, процессов, сети и БД.
    Обязательны все 13 столбцов
    из data/README.md, включая order_purchase_timestamp. Оставьте функцию
    доступной для прямой проверки на испорченных копиях файла.
    """
    with Path(snapshot).open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
        missing = REQUIRED_COLUMNS.difference(header)
        if missing:
            raise ValueError(f"Snapshot is missing required columns: {', '.join(sorted(missing))}")
        if len(header) != len(set(header)):
            raise ValueError("Snapshot column names must be unique")
        stream.seek(0)
        frame = _read_csv(stream)

    if frame.empty:
        raise ValueError("Snapshot must contain at least one order")
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("Snapshot rows must match the CSV header")
    order_ids = frame["order_id"].str.strip()
    if order_ids.isna().any() or order_ids.eq("").any() or not order_ids.is_unique:
        raise ValueError("Snapshot order_id values must be nonempty and unique")
    if not np.isfinite(frame[list(NUMERIC_COLUMNS)].to_numpy()).all():
        raise ValueError("Snapshot numeric features must be finite")
    if not frame["late"].isin([0, 1]).all():
        raise ValueError("Snapshot late must contain binary numeric labels (0 or 1)")
    dates = pd.to_datetime(frame["order_purchase_timestamp"], format="ISO8601", errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("Snapshot order_purchase_timestamp must contain valid timestamps")
    return str(snapshot)


def prepare(snapshot):
    """Разделить данные по времени, вернуть пути и метаданные для train.

    Сохраните путь и SHA256 байтов исходного snapshot, не подготовленных таблиц.
    Внутренний формат prepared выбираете сами: обязательного набора ключей нет.
    """
    from olist_model import split_by_time

    snapshot = Path(snapshot).resolve()
    cutoff = os.environ.get("OLIST_CUTOFF", "2017-07-31")
    with snapshot.open("rb") as stream:
        snapshot_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        frame = _read_csv(stream)
    train_frame, validation_frame = split_by_time(frame, cutoff=cutoff)

    train_path = snapshot.with_name("train.csv")
    validation_path = snapshot.with_name("validation.csv")
    with TemporaryDirectory(dir=snapshot.parent) as directory:
        staged_train = Path(directory) / train_path.name
        staged_validation = Path(directory) / validation_path.name
        train_frame.to_csv(staged_train, index=False, encoding="utf-8")
        validation_frame.to_csv(staged_validation, index=False, encoding="utf-8")
        staged_train.replace(train_path)
        staged_validation.replace(validation_path)

    return {
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_sha256,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "cutoff": cutoff,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
    }


def train(prepared, *, dag_run_id, epochs=None, seed=None):
    """Обучить модель и сохранить параметры, метрики и полную модель в MLflow.

    DAG передаёт dag_run_id из фактического run_id контекста задачи Airflow.
    Запишите его без изменений в тег dag_run_id, а SHA256 исходного файла extract
    из prepared — в параметр snapshot_sha256. Верните идентификатор MLflow run.
    Остальные имена параметров: epochs, seed, lr, cutoff, train_rows, validation_rows;
    метрики по эпохам: train_loss и validation_pr_auc. Полная модель: model/.
    epochs/seed — необязательный бонус для Params или dag_run.conf; при None
    используйте рабочие значения по умолчанию. Эту сигнатуру можно адаптировать
    вместе с DAG, сохраняя обязательные метаданные и публичную validate(snapshot).
    """
    import mlflow
    from olist_model import fit_model, save_predictor

    epochs = int(os.environ.get("OLIST_EPOCHS", "5")) if epochs is None else epochs
    seed = int(os.environ.get("OLIST_SEED", "42")) if seed is None else seed
    lr = 0.003
    train_frame = _read_csv(prepared["train_path"])
    validation_frame = _read_csv(prepared["validation_path"])

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("olist_training")
    with mlflow.start_run(tags={"dag_run_id": dag_run_id, "snapshot_path": prepared["snapshot"]}) as run:
        mlflow.log_params({
            "epochs": epochs,
            "seed": seed,
            "lr": lr,
            "cutoff": prepared["cutoff"],
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "snapshot_sha256": prepared["snapshot_sha256"],
        })
        predictor, history = fit_model(train_frame, validation_frame, epochs=epochs, seed=seed, lr=lr)
        for entry in history:
            mlflow.log_metrics(
                {"train_loss": entry["train_loss"], "validation_pr_auc": entry["validation_pr_auc"]},
                step=entry["epoch"],
            )

        run_id = run.info.run_id
        model_path = Path(prepared["snapshot"]).parent / "models" / run_id
        save_predictor(predictor, model_path)
        mlflow.log_artifacts(str(model_path), artifact_path="model")
    return run_id
