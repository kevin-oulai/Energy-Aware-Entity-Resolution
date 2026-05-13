import pandas as pd
import numpy as np
import os
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable
    
from utils.utils import convert_token_value, data_cleaning


IDS_DIR = os.path.join("data", "ids")

def sequence_generating_m1(df):
    """
    Generate comparable sequences for llm
    label(s) value: 0 or 1
    text_n value: Concatenate the property name and property value (ex. title-title_value_1 name-name_value_1)

    :param df: [_id, label, table1.id1, table2.id, table1.title, table2.title, table1.name, table2.name, ...]
    :return df_result: [text1, text2, labels]
    """
    df_result = pd.DataFrame(columns=["text1", "text2", "labels"])
    for _, df_row in tqdm(df.iterrows(), total=len(df), desc="# Reading data"):
        text1 = ""
        text2 = ""
        labels = 0
        for col in df.columns:
            col_name_list = col.split(".")
            if col == "label":
                labels = df_row[col]
            elif len(col_name_list) > 1 and col_name_list[1] != "id":
                if int(col_name_list[0][-1]) == 1:
                    text1 = text1 + str(col_name_list[1]) + str(df_row[col])
                elif int(col_name_list[0][-1]) == 2:
                    text2 = text2 + str(col_name_list[1]) + str(df_row[col])
        text1 = data_cleaning(text1)
        text2 = data_cleaning(text2)
        df_result.loc[len(df_result)] = [text1, text2, labels]
    return df_result


def clear_id_counters(config=None):
    if not os.path.isdir(IDS_DIR):
        return

    version_name = None
    if isinstance(config, dict):
        version_name = str(config.get("version_name", "")).strip()

    for file_name in os.listdir(IDS_DIR):
        file_path = os.path.join(IDS_DIR, file_name)
        if not os.path.isfile(file_path):
            continue
        if version_name and not file_name.startswith(f"{version_name}_"):
            continue
        os.remove(file_path)

    if not os.listdir(IDS_DIR):
        os.rmdir(IDS_DIR)


def _resolve_id_config(config, is_training=False):
    version_name = str(config.get("version_name", "default")).strip() or "default"

    if is_training:
        return {
            "counter_path": os.path.join(IDS_DIR, f"{version_name}_A.txt"),
            "prefix": "A_",
        }

    return {
        "counter_path": os.path.join(IDS_DIR, f"{version_name}_B.txt"),
        "prefix": "B_",
    }


def _load_counter(counter_path):
    if not os.path.exists(counter_path):
        return 0
    with open(counter_path, "r", encoding="utf-8") as fp:
        value = fp.read().strip()
    return int(value) if value else 0


def _save_counter(counter_path, counter_value):
    with open(counter_path, "w", encoding="utf-8") as fp:
        fp.write(str(counter_value))


def _generate_incremental_rids(size, start_counter, prefix):
    return [f"{prefix}{counter}" for counter in range(start_counter, start_counter + size)]


def _drop_source_id_columns(raw_data):
    id_columns = [col for col in raw_data.columns if str(col).strip().lower() == "id"]
    if id_columns:
        raw_data = raw_data.drop(columns=id_columns)
    return raw_data



def index_normalization(config, raw_data, raw_data_path, is_training):
    """
    incremental mode: index and normalization
    * 'rid' should appear in meta_path if it is setted
    * note: change directly raw_data dataFrame - may be risky
    
    :param config: 
    :param raw_data: pd.DataFrame
    :return: pd.DataFrame
    """
    # ===== index =====
    if raw_data is None:
        raw_data = pd.read_csv(raw_data_path)
    else:
        raw_data = raw_data.copy()

    raw_data = _drop_source_id_columns(raw_data)

    os.makedirs(IDS_DIR, exist_ok=True)
    id_config = _resolve_id_config(config, is_training)
    counter_path = id_config["counter_path"]
    prefix = id_config["prefix"]
    current_counter = _load_counter(counter_path)
    print(f'# current counter {current_counter}, raw data len {len(raw_data)}, prefix {prefix}')
    raw_data["rid"] = _generate_incremental_rids(len(raw_data), current_counter, prefix)
    _save_counter(counter_path, current_counter + len(raw_data))

    if "rid" not in raw_data.columns:
        raise ValueError("Failed to generate 'rid' column during normalization.")

    meta_path = config.get("meta_path", [])
    result = None
    if meta_path:
        # with meta_path 
        paths = meta_path if isinstance(meta_path[0], list) else [meta_path]

        meta_node = set()
        for path in paths:
            meta_node.update(path)
        cols = [col for col in raw_data.columns if col in meta_node]
    else:
        # without meta_path 
        cols = [col for col in raw_data.columns]
    
    # ===== normalization ===== 
    # mapping
    col_to_idx = {col: idx for idx, col in enumerate(cols)}

    # Convert to a list of tuples
    data_tuples = list(raw_data[cols].itertuples(index=False, name=None))
    result_rows = []
    for row_data in tqdm(data_tuples, total=len(raw_data), desc="# Indexing/Normalizing meta_path"):
        row_result = {}

        for col in cols:
            col_idx = col_to_idx[col]
            cell_value = row_data[col_idx]

            if col != 'rid':
                token_list, _ = convert_token_value(cell_value)
            else:
                token_list = [cell_value]
          
            row_result[col] = token_list

        result_rows.append(row_result)

    result = pd.DataFrame(result_rows, columns=cols)
    return result