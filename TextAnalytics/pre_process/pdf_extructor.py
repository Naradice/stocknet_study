class Point:
    __edges_dict = {8: "left", 4: "right", 2: "top", 1: "bottom", 10: "left_top", 9: "left_bottom", 6: "right_top", 5: "right_bottom"}

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        # 0:left, 1: right, 2: top, 3: bottom
        self.__lines = [None, None, None, None]

    @property
    def left(self):
        return self.__lines[0]

    @left.setter
    def left(self, value):
        self.__lines[0] = value

    @property
    def right(self):
        return self.__lines[1]

    @right.setter
    def right(self, value):
        self.__lines[1] = value

    @property
    def top(self):
        return self.__lines[2]

    @top.setter
    def top(self, value):
        self.__lines[2] = value

    @property
    def bottom(self):
        return self.__lines[3]

    @bottom.setter
    def bottom(self, value):
        self.__lines[3] = value

    def has_left(self):
        return self.__lines[0] is not None

    def has_right(self):
        return self.__lines[1] is not None

    def has_top(self):
        return self.__lines[2] is not None

    def has_bottom(self):
        return self.__lines[3] is not None

    def get_edge_type(self):
        b = 1 if self.__lines[0] is None else 0
        for t in self.__lines[1:]:
            b = b << 1
            if t is None:
                b += 1
        return b

    def edge_type_to_str(self, value):
        if value in self.__edges_dict:
            return self.__edges_dict[value]
        else:
            return None

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


def __get_right_line(current_point, right_point, horizontal_lines, mergin):
    candidates = horizontal_lines[(abs(horizontal_lines.y_min - current_point.y) <= mergin)]
    candidates = candidates[(candidates.x_min - current_point.x < mergin) & (candidates.x_max - right_point.x > -mergin)]
    candidates = candidates.sort_values(by="x_max", ascending=True)
    right_lines = candidates.iloc[0:1]
    length = len(right_lines)
    if length == 1:
        return right_lines.iloc[0]
    elif length > 1:
        # print(f"found multiple lines on {current_point} for right side")
        return right_lines.iloc[0]
    else:
        return None


def __get_left_line(current_point, left_point, horizontal_lines, mergin):
    candidates = horizontal_lines[(abs(horizontal_lines.y_min - current_point.y) <= mergin)]
    candidates = candidates[(candidates.x_max - current_point.x > -mergin) & (candidates.x_min - left_point.x < mergin)]
    candidates = candidates.sort_values(by="x_max", ascending=True)
    length = len(candidates)
    if length == 1:
        return candidates.iloc[0]
    elif length > 1:
        # print(f"found multiple lines on {current_point} for left side")
        return candidates.iloc[0]
    else:
        return None


def __get_bottom_line(current_point, bottom_point, vertical_lines, mergin):
    candidates = vertical_lines[(abs(vertical_lines.x_min - current_point.x) <= mergin)]
    candidates = candidates[(candidates.y_max - current_point.y > -mergin) & (candidates.y_min - bottom_point.y > -mergin)]
    candidates = candidates.sort_values(by="x_max", ascending=True)
    length = len(candidates)
    if length == 1:
        return candidates.iloc[0]
    elif length > 1:
        # print(f"found multiple lines on {current_point} for bottom side")
        return candidates.iloc[0]
    else:
        return None


def __get_top_line(current_point, top_point, vertical_lines, mergin):
    candidates = vertical_lines[(abs(vertical_lines.x_min - current_point.x) <= mergin)]
    candidates = candidates[(current_point.y - candidates.y_min > -mergin) & (candidates.y_max - top_point.y > -mergin)]
    candidates = candidates.sort_values(by="x_max", ascending=True)
    length = len(candidates)
    if length == 1:
        return candidates.iloc[0]
    elif length > 1:
        # print(f"found multiple lines on {current_point} for right side")
        return candidates.iloc[0]
    else:
        return None
