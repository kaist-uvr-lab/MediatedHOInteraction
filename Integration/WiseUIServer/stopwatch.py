from datetime import datetime
from time import time

class StopWatch:
    def __init__(self, name=None):
        self.reset()

    def reset(self):
        self.start_datetime = None
        self.end_datetime = None

        self.start_time = None
        self.end_time = None

    def start(self):
        self.reset()
        self.start_datetime = datetime.now()
        self.start_time = time()

    def stop(self, name=None):
        self.end_datetime = datetime.now()
        self.end_time = time()
        #print(name, self.get_elapsed_seconds())
        import numpy as np
        spent_time = self.end_time - self.start_time + np.finfo(float).eps
        #print('{} : {}, {} fps'.format(name, spent_time, 1 / spent_time))
        return spent_time

    def get_elapsed_seconds(self):
        assert isinstance(self.start_datetime, datetime), 'call start() first'
        assert isinstance(self.end_datetime, datetime), 'call end() fist'
        # Return the total number of seconds contained in the duration
        # Equivalent to td / timedelta(seconds=1)
        return (self.end_datetime - self.start_datetime).total_seconds()