class Dataset:
    data: list[str]

    def __init__(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            self.data = [line.strip() for line in f.readlines() if line.strip()]  # remove empty lines

    def get_n_samples(self, n: int) -> list[str]:
        # return n samples from dataset, loops over if n > len(data)
        my_len = len(self.data)
        requested_n = n
        if requested_n <= my_len:
            return self.data[:requested_n]
        replications = (requested_n // my_len) + 1
        extended_data = self.data * replications
        return extended_data[:requested_n]

    @staticmethod
    def create_batches(items: list[str], batch_size: int) -> list[list[str]]:
        """Split a list of items into batches of the specified size"""
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batches.append(batch)
        return batches
