import os
import sys

import pandas as pd

module_path = os.path.abspath("../fprocess")
sys.path.append(module_path)

from fprocess import fprocess


def revert(dataset, values, ndx, is_tgt=False, columns=None):
    r_data = values
    indices = dataset.get_actual_index(ndx)
    if is_tgt:
        tgt_indices = []
        for __index in indices:
            ndx = dataset.output_indices(__index)
            tgt_indices.append(ndx.start)
        indices = tgt_indices
    # print(f"start revert procress for {[__process.kinds for __process in dataset.processes]}")
    for p_index in range(len(dataset.processes)):
        r_index = len(dataset.processes) - 1 - p_index
        process = dataset.processes[r_index]
        if hasattr(process, "revert_params"):
            # print(f"currently: {r_data[0, 0]}")
            params = process.revert_params
            if len(params) == 1:
                r_data = process.revert(r_data)
            else:
                params = {}
                if process.kinds == fprocess.MinMaxPreProcess.kinds:
                    r_data = process.revert(r_data, columns=columns)
                elif process.kinds == fprocess.SimpleColumnDiffPreProcess.kinds:
                    close_column = process.base_column
                    if p_index > 0:
                        processes = dataset.processes[:p_index]
                        required_length = [1]
                        base_processes = []
                        for base_process in processes:
                            if close_column in base_process.columns:
                                base_processes.append(base_process)
                                required_length.append(base_process.get_minimum_required_length())
                        if len(base_processes) > 0:
                            raise Exception("Not implemented yet")
                    base_indices = [index - 1 for index in indices]
                    base_values = dataset.org_data[close_column].iloc[base_indices]
                    r_data = process.revert(r_data, base_value=base_values)
                elif process.kinds == fprocess.DiffPreProcess.kinds:
                    if columns is None:
                        target_columns = process.columns
                    else:
                        target_columns = columns
                    if r_index > 0:
                        processes = dataset.processes[:r_index]
                        required_length = [process.get_minimum_required_length()]
                        base_processes = []
                        for base_process in processes:
                            if len(set(target_columns) & set(base_process.columns)) > 0:
                                base_processes.append(base_process)
                                required_length.append(base_process.get_minimum_required_length())
                        if len(base_processes) > 0:
                            required_length = max(required_length)
                            batch_base_indices = [index - required_length for index in indices]
                            batch_base_values = pd.DataFrame()
                            # print(f"  apply {[__process.kinds for __process in base_processes]} to revert diff")
                            for index in batch_base_indices:
                                target_data = dataset.org_data[target_columns].iloc[index : index + required_length]
                                for base_process in base_processes:
                                    target_data = base_process(target_data)
                                batch_base_values = pd.concat([batch_base_values, target_data.iloc[-1:]], axis=0)
                            batch_base_values = batch_base_values.values.reshape(1, *batch_base_values.shape)
                        else:
                            base_indices = [index - 1 for index in indices]
                            batch_base_values = dataset.org_data[target_columns].iloc[base_indices]
                    else:
                        base_indices = [index - 1 for index in indices]
                        batch_base_values = dataset.org_data[target_columns].iloc[base_indices].values
                    r_data = process.revert(r_data, base_values=batch_base_values, columns=columns)
                else:
                    raise Exception(f"Not implemented: {process.kinds}")
    return r_data
