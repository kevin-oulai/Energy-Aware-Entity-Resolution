import argparse
import json
import os
import sys
from time import time

import pandas as pd
from pandas.errors import EmptyDataError
from pandas import DataFrame
from ruamel.yaml import YAML
from utils.buffers import load_earliest_buffer, write_buffer, get_earliest_window_index, wait_for_buffer, write_eos, delete_earliest_buffer_file
from utils.codecarbon import ccdecorator

from pipeline.normalization import index_normalization, sequence_generating_m1

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
BUFFER_PATH = "/app/data/buffers/"


def _log(message: str):
    print(message, file=sys.stderr)

def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def safe_read_csv(path):
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _counter_path_for_version(version_name: str) -> str:
    save_dir = os.path.join("data", "ids")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{version_name}.txt")


def _maybe_reset_rid_counter(config: dict, force = False) -> str:
    norm_cfg = config.get("normalization", {}) if isinstance(config, dict) else {}
    norm_cfg = norm_cfg if isinstance(norm_cfg, dict) else {}
    reset_counter = _as_bool(norm_cfg.get("reset_counter_on_start", False), default=False)
    version_name = str(config.get("version_name", "test"))
    counter_path = _counter_path_for_version(version_name)

    if reset_counter or force:
        with open(counter_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("0")

    return counter_path


def _resolve_embedding_raw_df(config: dict, raw_data: dict | DataFrame):
    if isinstance(raw_data, dict):
        if isinstance(raw_data.get("data"), pd.DataFrame):
            return raw_data["data"]
        payload_df = next((value for value in raw_data.values() if isinstance(value, pd.DataFrame)), pd.DataFrame())
        if not payload_df.empty:
            return payload_df

    if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
        return raw_data

    source_a = config.get("data_source_A")
    source_b = config.get("data_source_B")

    # For ER evaluation datasets (e.g. Fodors/Zagat), ground-truth ids assume
    # A and B are indexed in one continuous sequence.
    if source_a and source_b:
        df_a = safe_read_csv(source_a)
        df_b = safe_read_csv(source_b)
        if not df_a.empty and not df_b.empty:
            return pd.concat([df_a, df_b], ignore_index=True)

    if isinstance(raw_data, dict):
        if isinstance(raw_data.get("data"), pd.DataFrame):
            return raw_data["data"]
        return next((value for value in raw_data.values() if isinstance(value, pd.DataFrame)), pd.DataFrame())

    if isinstance(raw_data, pd.DataFrame):
        return raw_data

    return pd.DataFrame()


def serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


def _sample_records(df: pd.DataFrame, limit: int = 3):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    columns = [col for col in ("rid",) if col in df.columns]
    if not columns:
        columns = list(df.columns[: min(len(df.columns), 5)])
    return df.loc[:, columns].head(limit).to_dict(orient="records")

# @ccdecorator
def normalization(config: dict, raw_data: dict | DataFrame, is_training: bool = False):
    _log(
        f"[normalization_distribution] mode={config.get('mode')} is_training={is_training} "
        f"raw_type={type(raw_data).__name__}"
    )
    if 'embedding' in config.get('mode', ''):
        # in incremental mode, we index records and normalize

        raw_data_path = config.get("data_source_A")
        raw_df = _resolve_embedding_raw_df(config, raw_data)
        _log(
            f"[normalization_distribution] resolved_raw rows={len(raw_df)} cols={list(raw_df.columns)[:12]} "
            f"sample={_sample_records(raw_df)}"
        )
        _maybe_reset_rid_counter(config)
        processed_data = index_normalization(config, raw_df, raw_data_path)
        if isinstance(processed_data, pd.DataFrame):
            _log(
                f"[normalization_distribution] processed rows={len(processed_data)} cols={list(processed_data.columns)[:12]} "
                f"sample={_sample_records(processed_data)}"
            )
    else:
        if "bert" in config.get("mode", ""):
            if "training" in config.get("mode", ""):
                raw_data = {
                    "train": safe_read_csv(config.get("trainset_path")),
                    "eval": safe_read_csv(config.get("evalset_path")),
                }
            else:
                raw_data = {
                    "test": safe_read_csv(config.get("testset_path")),
                }

        for key, df in raw_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                raw_data[key] = sequence_generating_m1(df)
        processed_data = raw_data
    return processed_data


def load_raw_data(raw_data_value: str) -> dict | DataFrame:
    if os.path.isfile(raw_data_value) and raw_data_value.lower().endswith(".csv"):
        return {"data": pd.read_csv(raw_data_value)}
    return {"data": pd.DataFrame([{"value": raw_data_value}])}

def _exit(output_path=None, output=None, output_buffer_path=None):
    if output_buffer_path:
        write_eos(output_buffer_path, reason=f"timeout_no_initial_buffer")
    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)

def run_argo_batch(mode: str,raw_data_value: str, data_source_a: str = "", data_source_b: str = "",):
    config = load_config()
    config["mode"] = mode
    if isinstance(data_source_a, str) and data_source_a.strip():
        config["data_source_A"] = data_source_a.strip()
    if isinstance(data_source_b, str) and data_source_b.strip():
        config["data_source_B"] = data_source_b.strip()
    is_training = "training" in mode
    output = normalization(config=config, raw_data=load_raw_data(raw_data_value), is_training=is_training)
    _exit(output=output)

def run_argo_incremental():
    config = load_config()

    load_buffer_path = BUFFER_PATH + "raw_data"
    print(f"[normalization_distribution] waiting for raw buffer at {load_buffer_path}", file=sys.stderr)
    first_ready = wait_for_buffer(load_buffer_path, timeout_seconds=120)
    if first_ready is None:
        _exit(output_buffer_path=BUFFER_PATH + "processed_data")
        _exit(output_buffer_path=BUFFER_PATH + "processed_data_feature")
        return
    _maybe_reset_rid_counter(config, force=True)
    raw_data = load_earliest_buffer(load_buffer_path)
    while raw_data is not None:
        if raw_data.empty:
            print(f"[normalization_distribution] empty raw buffer at {load_buffer_path}; waiting for next window", file=sys.stderr)
            next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
            raw_data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
            continue
        print(
            f"[normalization_distribution] window_index={get_earliest_window_index(load_buffer_path)} "
            f"raw_type={type(raw_data).__name__} rows={len(raw_data)}",
            file=sys.stderr,
        )
        returned = normalization(config=config, raw_data=raw_data, is_training=False)
        window_index = get_earliest_window_index(BUFFER_PATH + "raw_data")
        if returned is not None:
            if isinstance(returned, pd.DataFrame):
                print(
                    f"[normalization_distribution] writing window={window_index} rows={len(returned)} "
                    f"sample={_sample_records(returned)}",
                    file=sys.stderr,
                )
            write_buffer(returned, BUFFER_PATH + "processed_data", window_index, extension="csv")
            write_buffer(returned, BUFFER_PATH + "processed_data_feature", window_index, extension="csv")

        delete_earliest_buffer_file(load_buffer_path)
        next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
        raw_data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
    _exit(output_buffer_path=BUFFER_PATH + "processed_data")
    _exit(output_buffer_path=BUFFER_PATH + "processed_data_feature")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalization distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--raw_data", default="")
    parser.add_argument("--data_source_A", default="")
    parser.add_argument("--data_source_B", default="")
    args = parser.parse_args()
    if "inc" in args.mode:
        run_argo_incremental()
    else:
        run_argo_batch(
            mode=args.mode,
            raw_data_value=args.raw_data,
            data_source_a=args.data_source_A,
            data_source_b=args.data_source_B,
        )