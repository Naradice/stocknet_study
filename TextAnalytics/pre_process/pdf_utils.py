import re

import numpy as np
import pandas as pd
from pdfminer.layout import LTLine
from pdfminer.layout import LTTextContainer
from pdfminer.layout import LTRect


BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]


def is_near_point(p1, p2, mergin=1.0):
    if abs(p1 - p2) <= mergin:
        return True
    else:
        return False


def is_adj(base_node, target_node):
    x_l, y_l, x_r, y_t = base_node
    xt_l, yt_l, xt_r, yt_t = target_node
    height = abs(yt_t - yt_l)
    width = abs(xt_r - xt_l)
    # indicates (edge_type, bbox). If target_node is connected to base_node, bbox is returned.
    default_response = (-1, None)
    # edge types: 0-> left, 1-> right, 2-> top, 3-> buttom
    is_horizontal_line = abs(x_r - x_l) > 0
    is_vertical_line = abs(y_t - y_l) > 0

    if height > 0 and width > 0:
        print(f"unexpected target bbox. It is not a line: {target_node}")
        return default_response
    if is_vertical_line is True and is_horizontal_line is True:
        print(f"unexpected base bbox. It is not a line: {target_node}")
        return default_response
    if height > 0:
        # we can check eigher left or right since width is zero
        # TODO: check if boarder width is related to coordinates
        if is_horizontal_line:
            if is_near_point(y_t, yt_t):
                if is_near_point(x_l, xt_l):
                    return (0, target_node)
                elif is_near_point(x_r, xt_r):
                    return (1, target_node)
                elif (x_l < xt_l) and (x_r > xt_l):
                    return (-1, target_node)
                else:
                    # if there are multiple table or merged cell, this case happens
                    return default_response
            else:
                return default_response
        else:
            if is_near_point(x_l, xt_l):
                if is_near_point(y_t, yt_l):
                    return (2, target_node)
                elif is_near_point(y_l, yt_t):
                    return (3, target_node)
                elif (y_l < yt_l) and (y_t > yt_t):
                    return (-1, target_node)
                else:
                    return default_response
            else:
                return default_response
    elif width > 0:
        if is_vertical_line:
            if is_near_point(x_l, xt_l):
                if is_near_point(y_t, yt_t):
                    return (2, target_node)
                elif is_near_point(y_l, yt_l):
                    return (3, target_node)
                elif (y_l < yt_l) and (y_t > yt_t):
                    return (-1, target_node)
                else:
                    return default_response
            else:
                return default_response
        else:
            if is_near_point(y_t, yt_t):
                if is_near_point(x_l, xt_r):
                    return (0, target_node)
                elif is_near_point(x_r, xt_l):
                    return (1, target_node)
                elif (x_l < xt_l) and (x_r > xt_l):
                    return (-1, target_node)
                else:
                    return default_response
            else:
                return default_response
    else:
        print(f"target line seems to be a point, not a line: {target_node}")
        return default_response


def separate_adjs(target_node, remaining_nodes):
    adjs = []
    others = []
    for index in remaining_nodes.index:
        item = remaining_nodes.loc[index]
        edge_type, coord = is_adj(target_node, item)
        if coord is None:
            others.append(item)
        else:
            adjs.append(item)
    return adjs, pd.DataFrame(others)


def mergin_filter(unique_array: np.ndarray, mergin):
    filtered = []
    values = unique_array.copy()
    while len(values) > 1:
        value = values[0]
        values = values[1:]
        if True in (abs(values - value) < mergin):
            continue
        filtered.append(value)
    if len(values) > 0:
        value = values[0]
        filtered.append(value)
    return np.array(filtered)


def is_target_overwrap_to_bottom(base_bb, target_bb):
    x_min, y_min, x_max, y_max = base_bb
    xt_min, yt_min, xt_max, yt_max = target_bb

    if (x_min <= xt_min) and (x_max >= xt_max) and (yt_max > y_min) and (y_max - yt_max >= 0):
        return True
    else:
        return False


def is_target_overwrap_to_right(base_bb, target_bb):
    x_min, y_min, x_max, y_max = base_bb
    xt_min, yt_min, xt_max, yt_max = target_bb

    if (y_min <= yt_min) and (y_max >= yt_max) and (x_min - xt_min >= 0) and (xt_max < xt_min):
        return True
    else:
        return False


def is_target_in_top(base_bb, target_bb):
    x_min, y_min, x_max, y_max = base_bb
    xt_min, yt_min, xt_max, yt_max = target_bb

    if y_min <= yt_min:
        return True
    else:
        return False


def tables_lines_to_bbox(table_lines: pd.DataFrame):
    return (table_lines.x_min.min(), table_lines.y_min.min(), table_lines.x_max.max(), table_lines.y_max.max())


def is_in_bb(target_out_bb, target_in_bb):
    x_min, y_min, x_max, y_max = target_out_bb
    xt_min, yt_min, xt_max, yt_max = target_in_bb

    if (x_min <= xt_min) and (y_min <= yt_min) and (x_max >= xt_max) and (y_max >= yt_max):
        return True
    else:
        return False


def texts_ele_to_df(texts_ele) -> pd.DataFrame:
    items = []
    for text_ele in texts_ele:
        items.append([*text_ele.bbox, text_ele.get_text()])
    return pd.DataFrame(items, columns=BBOX_COLUMNS)


def retrieve_table_elements(page) -> pd.DataFrame:
    lines = []
    _rects = []

    for item in page:
        if isinstance(item, LTLine):
            lines.append(item.bbox)
        elif isinstance(item, LTRect):
            _rects.append(item)

    if len(_rects) >= 2:
        # splitt rects to lines
        for rect in _rects:
            x_min, y_min, x_max, y_max = rect.bbox
            left_line = (x_min, y_min, x_min, y_max)
            right_line = (x_max, y_min, x_max, y_max)
            top_line = (x_min, y_max, x_max, y_max)
            bottom_line = (x_min, y_min, x_max, y_min)
            lines.extend([left_line, right_line, top_line, bottom_line])
    lines = pd.DataFrame(lines, columns=BBOX_COLUMNS)
    lines = lines.sort_values(by=BBOX_COLUMNS[3], ascending=False)
    lines.reset_index(inplace=True, drop=True)
    return lines


def get_line_count(text, new_line="\n"):
    sentences = text.split(new_line)

    lines_count = 0
    for sentence in sentences:
        sentence = re.sub(r"[ 　]", "", sentence)
        if len(sentence) > 0:
            lines_count += 1
    return lines_count


def get_one_line_height(text_item, new_line="\n"):
    if isinstance(text_item, LTTextContainer) is True:
        _, y_min, _, y_max = text_item.bbox
        text = text_item.get_text()
    elif isinstance(text_item, pd.Series) is True:
        _, y_min, _, y_max = text_item[["x_min", "y_min", "x_max", "y_max"]]
        text = text_item["text"]
    else:
        print(f"unsupported types: {type(text_item)}")
        return None

    height = y_max - y_min
    lines_count = get_line_count(text, new_line)
    if lines_count > 1:
        height = height / lines_count
    return height


def get_threshold_of_newline(text_elements, new_line="\n"):
    if isinstance(text_elements, pd.DataFrame) is False:
        txt_df = texts_ele_to_df(text_elements)
    else:
        txt_df = text_elements
    txt_df.sort_values(by="y_min", ascending=False, inplace=True)

    line_diffs = []
    line_heights = []
    pre_y_min = None
    for index in txt_df.index:
        element = txt_df.loc[index]
        height = get_one_line_height(element, new_line)
        if height is not None:
            line_heights.append(height)
        if pre_y_min is None:
            pre_y_min = element["y_min"]
        else:
            y_max = element["y_max"]
            if y_max <= pre_y_min:
                current_line_diff = pre_y_min - y_max
                line_diffs.append(current_line_diff)
                pre_y_min = element["y_min"]
    if len(line_heights) > 0 and len(line_diffs) > 0:
        line_height = pd.Series(line_heights).mode()[0]
        line_diff = pd.Series(line_diffs).mode()[0]
        return line_height + line_diff
    return None


def rank_x_points(bbox_df):
    min_x_rank = bbox_df["x_min"].rank(method="dense").convert_dtypes(int)
    min_column_num = min_x_rank.max()
    h_alignment = "left"

    x_rank = bbox_df["x_max"].rank(method="dense").convert_dtypes(int)
    column_num = x_rank.max()
    if min_column_num > column_num:
        min_column_num = column_num
        h_alignment = "right"
        min_x_rank = x_rank

    x_rank = bbox_df[["x_min", "x_max"]].mean(axis=1).rank(method="dense").convert_dtypes(int)
    column_num = x_rank.max()
    if min_column_num > column_num:
        min_column_num = column_num
        h_alignment = "center"
        min_x_rank = x_rank
    if len(bbox_df) == min_column_num:
        h_alignment = "ambiguous"
    return min_x_rank, min_column_num, h_alignment


def rank_y_points(bbox_df):
    min_y_rank = bbox_df["y_min"].rank(method="dense").convert_dtypes(int)
    min_row_num = min_y_rank.max()
    v_alignment = "bottom"

    y_rank = bbox_df["y_max"].rank(method="dense").convert_dtypes(int)
    row_num = y_rank.max()
    if min_row_num > row_num:
        min_row_num = row_num
        v_alignment = "top"
        min_y_rank = y_rank

    y_rank = bbox_df[["y_min", "y_max"]].mean(axis=1).rank(method="dense").convert_dtypes(int)
    row_num = y_rank.max()
    if min_row_num > row_num:
        min_row_num = row_num
        v_alignment = "center"
        min_y_rank = y_rank
    if len(bbox_df) == min_row_num:
        v_alignment = "ambiguous"
    min_y_rank = abs(min_y_rank - min_row_num) + 1
    return min_y_rank, min_row_num, v_alignment


def h_overlap(bbox_df, bbox_df_alt):
    x_min, x_max = bbox_df[["x_min", "x_max"]]
    x_min_alt, x_max_alt = bbox_df_alt[["x_min", "x_max"]]
    if x_min <= x_min_alt and x_max >= x_min_alt:
        return True
    if x_min <= x_max_alt and x_max >= x_max_alt:
        return True
    if x_min_alt <= x_min and x_max_alt >= x_min:
        return True
    if x_min_alt <= x_max and x_max_alt >= x_max:
        return True
    return False


def v_overlap(bbox_df, bbox_df_alt):
    y_min, y_max = bbox_df[["y_min", "y_max"]]
    y_min_alt, y_max_alt = bbox_df_alt[["y_min", "y_max"]]
    if y_min <= y_min_alt and y_max >= y_min_alt:
        return True
    if y_min <= y_max_alt and y_max >= y_max_alt:
        return True
    if y_min_alt <= y_min and y_max_alt >= y_min:
        return True
    if y_min_alt <= y_max and y_max_alt >= y_max:
        return True
    return False


def get_text_sizes(text_df_with_xy: pd.DataFrame, row_num, column_num):
    column_sizes = []
    row_sizes = []

    for index in range(0, column_num):
        rows = text_df_with_xy[text_df_with_xy["x"] == index]
        if len(rows) == row_num:
            rows = rows.sort_values(by="y")
            current_row_sizes = rows[["y_min", "y_max"]].values
            if len(row_sizes) == 0:
                row_sizes = current_row_sizes
            else:
                row_sizes[row_sizes[:, 0] > current_row_sizes[:, 0]][:, 0] = current_row_sizes[:, 0]
                row_sizes[row_sizes[:, 1] < current_row_sizes[:, 1]][:, 1] = current_row_sizes[:, 1]

    for index in range(0, row_num):
        columns = text_df_with_xy[text_df_with_xy["y"] == index]
        if len(columns) == column_num:
            columns = columns.sort_values(by="x")
            current_column_sizes = columns[["x_min", "x_max"]].values
            if len(column_sizes) == 0:
                column_sizes = current_column_sizes
            else:
                column_sizes[:, 0] = np.where(column_sizes[:, 0] > current_column_sizes[:, 0], current_column_sizes[:, 0], column_sizes[:, 0])
                column_sizes[:, 1] = np.where([column_sizes[:, 1] < current_column_sizes[:, 1]], current_column_sizes[:, 1], column_sizes[:, 1])
    return row_sizes, column_sizes


def update_coords(text_df_with_xy: pd.DataFrame, row_sizes, column_sizes, mergin=0.5):
    if len(row_sizes) > 0 and len(column_sizes) > 0:
        print(column_sizes)
        # since text_df is already filtered by table bbox, put let end
        left_x = 0
        top_y = np.inf
        cells = []
        empty_cell = pd.DataFrame([[np.NaN for column in text_df_with_xy.columns]], columns=text_df_with_xy.columns)
        for y in range(1, len(row_sizes) + 1):
            if y < len(row_sizes):
                bottom_y = row_sizes[y, 1] + mergin
            else:
                bottom_y = 0
            row = text_df_with_xy[(text_df_with_xy["y_min"] >= bottom_y) & (text_df_with_xy["y_max"] <= top_y)]
            row.loc[:, "y"] = y - 1
            if y < len(row_sizes):
                top_y = row_sizes[y, 1]
            for x in range(1, len(column_sizes)):
                right_x = column_sizes[x, 0] - mergin
                cell = row[(row["x_min"] >= left_x) & (row["x_max"] <= right_x)]
                if len(cell) == 0:
                    cell = empty_cell.copy()
                    cell.loc[:, "y"] = y - 1
                cell.loc[:, "x"] = x - 1
                left_x = column_sizes[x, 0]
                cells.append(cell)
            right_x = np.inf
            cell = row[(row["x_min"] >= left_x) & (row["x_max"] <= right_x)]
            if len(cell) == 0:
                cell = empty_cell.copy()
                cell.loc[:, "y"] = y - 1
            cell.loc[:, "x"] = x
            cells.append(cell)
            left_x = 0

        df = pd.concat(cells)
        return df
    return pd.DataFrame()


def add_coords(text_df: pd.DataFrame):
    df = text_df.sort_values(by="x_min")
    target_text = df.iloc[0]
    x = 0
    x_coords = [x]
    for index in range(1, len(df)):
        next_text = df.iloc[index]
        if h_overlap(target_text, next_text) is True:
            x_coords.append(x)
        else:
            x += 1
            x_coords.append(x)
            target_text = next_text
    df["x"] = x_coords

    df = df.sort_values(by="y_min", ascending=False)
    target_text = df.iloc[0]
    y = 0
    y_coords = [y]
    for index in range(1, len(df)):
        next_text = df.iloc[index]
        if v_overlap(target_text, next_text) is True:
            y_coords.append(y)
        else:
            y += 1
            y_coords.append(y)
            target_text = next_text
    df["y"] = y_coords
    row_num = df["x"].value_counts().max()
    column_num = df["y"].value_counts().max()
    row_sizes, column_sizes = get_text_sizes(df, row_num, column_num)
    df = update_coords(df, row_sizes, column_sizes)
    return df, row_num, column_num


def create_table(text_df):
    text_df, row_num, column_num = add_coords(text_df)
    if column_num > 1 and row_num > 1:
        if len(pd.unique(text_df["x"].value_counts())) == 1:
            tables = []
            for column_index in range(0, column_num):
                items = text_df[(text_df["x"] == column_index)]
                rows = []
                for y_index in range(0, row_num):
                    if y_index in items["y"].values:
                        cell = items[items["y"] == y_index]["text"].iloc[0]
                    else:
                        cell = None
                    rows.append(cell)
                tables.append(rows)
            tables = pd.DataFrame(tables).T
            return tables
        else:
            tables = []
            text_df = text_df.sort_values(by="y", ascending=False)
            for row_index in range(0, row_num):
                items = text_df[(text_df["y"] == row_index)]
                items = items.sort_values(by="x")
                rows = [text for text in items["text"]]
                tables.append(rows)
            tables = pd.DataFrame(tables)
            return tables
    else:
        return None


def retrieve_table_lines(target_lines: pd.DataFrame):
    initial_node = target_lines.iloc[0]
    table_nodes = []
    adjs = [initial_node]
    remaining_nodes = target_lines.iloc[1:].copy()
    while len(remaining_nodes) > 0:
        pre_length = len(remaining_nodes)
        if len(adjs) > 0:
            new_adjs = []
            for adj in adjs:
                children, remaining_nodes = separate_adjs(adj, remaining_nodes)
                new_adjs.extend(children)
                table_nodes.append(adj)
            adjs = new_adjs
        else:
            # print("no adjs")
            break
        if pre_length == len(remaining_nodes):
            # print("Stopped since remaining nodes was not changed")
            break
    table_nodes = pd.DataFrame(table_nodes)
    remaining_adjs = pd.DataFrame(adjs)
    if len(table_nodes) > 0:
        table_nodes = pd.concat([table_nodes, remaining_adjs])
    else:
        remaining_nodes = pd.concat([remaining_adjs, remaining_nodes])
    table_nodes.reset_index(inplace=True, drop=True)
    return table_nodes, remaining_nodes


def create_table_from_lines(lines, txt_elements, mergin, texts_num_in_oneline, default_newline_threshold):
    table_array = []
    data_index = 0
    while data_index < len(lines):
        df, bb = lines[data_index]
        if data_index > 0:
            _, pre_bb = lines[data_index - 1]
            left_top = (pre_bb[0], pre_bb[3])
            right_bottom = (bb[2], bb[1])
            candidate_bbox = (left_top[0], right_bottom[1], right_bottom[0], left_top[1])
        else:
            candidate_bbox = bb
        candidate_txts = []
        remaining_txts = []
        for text in txt_elements:
            if is_in_bb(candidate_bbox, text.bbox):
                candidate_txts.append(text)
            else:
                if candidate_bbox[0] - mergin <= text.bbox[0] and text.bbox[2] <= candidate_bbox[2] + mergin:
                    remaining_txts.append(text)
        candidate_txts = texts_ele_to_df(candidate_txts)
        remaining_txts = texts_ele_to_df(remaining_txts)
        remaining_txts.sort_values(by="y_min", ascending=False, inplace=True)
        # Add below candidates
        below_txts = remaining_txts[(remaining_txts["y_max"] <= candidate_bbox[1])]
        if data_index + 1 < len(lines):
            _, next_bb = lines[data_index + 1]
            below_txts = below_txts[below_txts["y_min"] >= next_bb[1]]
        if len(candidate_txts) >= texts_num_in_oneline * 2:
            newline_threshold = get_threshold_of_newline(candidate_txts)
            if newline_threshold is None:
                newline_threshold = default_newline_threshold
        else:
            newline_threshold = default_newline_threshold

        new_line_exists_in_below = False
        if len(below_txts) > texts_num_in_oneline:
            unique_y_points = pd.unique(below_txts["y_min"])
            pre_y_min = candidate_bbox[1]
            for y_point in unique_y_points:
                element_df = below_txts[below_txts["y_min"] == y_point]
                if len(element_df) > texts_num_in_oneline:
                    # assume this is cell
                    pre_y_min = y_point
                    continue
                text = element_df["text"].iloc[0]
                line_count = get_line_count(text)
                diff = pre_y_min - y_point
                if line_count > 0:
                    diff = diff / line_count
                if diff > newline_threshold:
                    new_line_exists_in_below = True
                    break
                pre_y_min = y_point
            below_candidates = below_txts[below_txts["y_min"] >= pre_y_min]
            candidate_txts = pd.concat([candidate_txts, below_candidates])
        # Add above candidates
        above_txts = remaining_txts[remaining_txts["y_min"] >= candidate_bbox[3]]
        if data_index > 1:
            _, upper_bb = lines[data_index - 2]
            above_txts = above_txts[above_txts["y_max"] <= upper_bb[1]]
        if pre_table is None and len(above_txts) > texts_num_in_oneline:
            pre_y_max = candidate_bbox[3]

            above_y_points = pd.unique(above_txts["y_max"])
            for y_point in reversed(above_y_points):
                element_df = above_txts[above_txts["y_max"] == y_point]
                if len(element_df) > texts_num_in_oneline:
                    # assume this is cell
                    pre_y_max = y_point
                    continue
                text = element_df["text"].max()
                line_count = get_line_count(text)
                diff = pre_y_max - y_point
                if line_count > 0:
                    diff = diff / line_count
                else:
                    # Above cell would be a header. If row has only empty value, handle it as
                    break
                if diff > newline_threshold:
                    break
                pre_y_min = y_point

            above_candidates = above_txts[above_txts["y_max"] <= pre_y_max]
            candidate_txts = pd.concat([above_candidates, candidate_txts])
        if len(candidate_txts) > texts_num_in_oneline:
            tables = create_table(candidate_txts)
            if tables is not None:
                if pre_table is not None:
                    tables = pd.concat([pre_table, tables])
                    pre_table = None
                if new_line_exists_in_below:
                    table_array.append(tables)
                else:
                    pre_table = tables
                data_index += 2
            else:
                if pre_table is not None:
                    table_array.append(pre_table)
                    pre_table = None
                data_index += 1
        else:
            data_index += 1

        if pre_table is not None:
            table_array.append(pre_table)
