import logging
import time
import threading
import curses
import signal


class CursesHandler(logging.Handler):
    def __init__(self, stdscr, input_box):
        super().__init__()
        self.stdscr = stdscr
        self.input_box = input_box
        self.log_lines = []
        self.max_lines = curses.LINES - 4

    def emit(self, record):
        if record is not None:
            log_entry = self.format(record)
            self.log_lines.append(log_entry)
        if len(self.log_lines) > self.max_lines:
            self.log_lines.pop(0)

        self.stdscr.clear()
        for idx, line in enumerate(self.log_lines):
            self.stdscr.addstr(idx, 0, line)
        self.stdscr.addstr(curses.LINES - 3, 0, "-" * curses.COLS)
        self.stdscr.addstr(
            curses.LINES - 1,
            0,
            f"input something: {self.input_box.get_current_input()}",
        )
        self.stdscr.refresh()


class InputBox:
    def __init__(self):
        self.current_input = ""

    def add_char(self, ch):
        if ch == curses.KEY_BACKSPACE or ch == 8:
            self.current_input = self.current_input[:-1]
        else:
            self.current_input += chr(ch)

    def get_current_input(self):
        return self.current_input

    def reset(self):
        self.current_input = ""


def sleep_messages(logger, stop_event):
    count = 0
    while count < 20:
        count += 1
        logger.info(f"test log: {count}")
        if stop_event.is_set():
            break
        time.sleep(1)


def main(stdscr):
    curses.echo(False)
    input_box = InputBox()
    stop_event = threading.Event()

    curses_handler = CursesHandler(stdscr, input_box)
    curses_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("output.log")
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger("CursesAndFileLogger")
    logger.addHandler(curses_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    t = threading.Thread(target=sleep_messages, args=(logger, stop_event), daemon=False)
    t.start()

    def signal_handler(sig, frame):
        stop_event.set()
        t.join()
        curses.endwin()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    user_input = None
    while user_input != "exit":
        ch = stdscr.getch(
            curses.LINES - 1,
            len("input something: ") + len(input_box.get_current_input()),
        )
        if ch == 10:
            user_input = input_box.get_current_input()
            stdscr.addstr(curses.LINES - 2, 0, f"You entered: {user_input}")
            stdscr.refresh()
            input_box.reset()
        else:
            input_box.add_char(ch)
        curses_handler.emit(None)
    signal_handler(None, None)


curses.wrapper(main)
