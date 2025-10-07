from time import perf_counter
import math
import functools

def disable_if_disabled(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.disabled:
            return
        return method(self, *args, **kwargs)
    return wrapper

class Chronometer:
    """
        A class used to measure and report the time taken by different tasks.

        Attributes:
            disabled (bool): Determines if the Chronometer is disabled. Default is False.
        """
    def __init__(self, disabled=False):
        self.counter = 0
        self.start_times = { }
        self.total_times = { }
        self.last_times = { }
        self.disabled = disabled

    @disable_if_disabled
    def new_cycle(self):
        """
        Starts or stops a full cycle, storing the total working time under the label "full_cycle".
        """
        call_time = perf_counter()
        if "full_cycle"  in self.start_times:
            elapsed_time = call_time - self.start_times["full_cycle"]
            self.total_times["full_cycle"] = self.total_times.get("full_cycle", 0) + elapsed_time
            self.last_times["full_cycle"] = elapsed_time
        self.start_times["full_cycle"] = call_time
        self.counter += 1

    @disable_if_disabled
    def start(self, label: str):
        """
        Starts measuring time for a task with a given label.

        Parameters:
            label (str): The label of the task.
        """
        self.start_times[label] = perf_counter()

    @disable_if_disabled
    def stop(self, label: str):
        """
        Stop measuring time for a task with a given label and stores the elapsed time.

        Parameters:
            label (str): The label of the task.
        """
        end_time = perf_counter()
        elapsed_time = end_time - self.start_times[label]
        self.total_times[label] = self.total_times.get(label, 0) + elapsed_time
        self.last_times[label] = elapsed_time

    @disable_if_disabled
    def print_stats(self):
        """
        Prints a table with mean and last time taken by each task
        """
        max_label_length = 10
        print()
        print(f"{'Label':<{max_label_length}} | Mean Time (ms) | Last Time (ms)")
        print("-" * (max_label_length + 2 + 16 + 2 + 15 ))

        mean_time_cycle = 1000 * self.total_times.get('full_cycle') / (self.counter - 1) if self.counter > 1 else math.nan
        last_time_cycle = self.last_times.get('full_cycle') * 1000 if self.counter > 1 else math.nan
        print(f"{'full_cycle':<{max_label_length}} | {mean_time_cycle:>13.2f} | {last_time_cycle:>13.2f}")

        for label, total_time in self.total_times.items():
            cropped_label = label[:max_label_length]
            if label == "full_cycle":
                continue
            else:
                mean_time = 1000 * total_time / self.counter
            last_time = self.last_times[label] * 1000
            print(f"{cropped_label:<{max_label_length}} | {mean_time:>13.2f} | {last_time:>13.2f}")
        print("-" * (max_label_length + 2 + 16 + 2 + 15 ))
        print(f"Cycles executed: {self.counter}")
        print()