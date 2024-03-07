from collections import defaultdict


class Averager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            if value is np.NDArray:
                self.total[key] = self.total[key] + value.sum()
                self.counter[key] = self.counter[key] + value.numel()
            else:
                self.total[key] = self.total[key] + value
                self.counter[key] = self.counter[key] + 1

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats
