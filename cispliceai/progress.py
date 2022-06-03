import sys

class ProgressOutput():
    def __init__(self, max_el, resolution=80):
        self.max_el = max_el
        self._resolution = resolution
        self._boxes = -1
        self.update(0)

    def update(self, progress: int):
        percent = progress / self.max_el if self.max_el > 0 else 1
        boxes = int(self._resolution * percent)
        if boxes != self._boxes:
            self._boxes = boxes
            dashes = self._resolution - boxes

            sys.stderr.write('\r' + '#' * boxes + '-' * dashes + ' (%d%%)' % (100 * percent))

        if progress == self.max_el:
            sys.stderr.write('\n')


if __name__ == '__main__':
    from time import sleep

    prog = ProgressOutput(400)
    for i in range(400):
        prog.update(i)
        sleep(0.5)