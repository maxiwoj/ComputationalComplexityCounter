from complexity_counter.testers import model


class TimeItResult:
    def __init__(self, computation_complexity, factors, base):
        self.computation_complexity = computation_complexity
        self.factors = factors
        self.base = base

    def time_predict(self, x):
        return model(self.factors, np.array([x]), self.base)[0]

    def max_complexity_predict(self, y):
        xmax = 10
        xmin = 0
        while y - self.time_predict(xmax) > 0:
            # print(y - self.time_predict(xmax))
            # print(str.format("xmax: {}, xmin: {}", xmax, xmin))
            xmax = xmax * 10
        est = int((xmax + xmin) / 2)
        while abs(y - self.time_predict(est)) > 0.0001 and xmax - xmin > 1:
            if (y - self.time_predict(est)) > 0:
                xmin = int((xmax + xmin) / 2)
            else:
                xmax = est
            est = int((xmax + xmin) / 2)
            # print(y - self.time_predict(est))
            # print(str.format("xmax: {}, xmin: {}, est: {}", xmax, xmin, est))
        return est