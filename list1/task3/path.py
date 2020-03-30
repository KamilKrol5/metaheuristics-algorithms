class Path(list):
    @property
    def cost(self) -> int:
        return len(self)

    def make_inversion(self, i, j) -> None:
        self[i], self[j] = self[j], self[i]

    def __hash__(self):
        return hash(tuple(self))
