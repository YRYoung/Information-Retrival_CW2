import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, all_dataframe: pd.DataFrame,
                 passages_tensors: dict[int, torch.Tensor],
                 queries_tensors: dict[int, torch.Tensor],
                 return_tensors: str,
                 passages_per_query=100,
                 generator=None, fake_tensor=False,
                 shuffle_passages=True, fixed_samples=False, auto_amend=True):
        super(Dataset).__init__()
        self.auto_amend = auto_amend
        self.shuffle_passages = shuffle_passages
        self.fake_tensor = fake_tensor
        self.return_tensors = return_tensors

        self.generator = generator

        self.all_dataframe = [[value.reset_index(drop=True, inplace=True), value][1] for key, value in
                              all_dataframe.groupby('q_idx', sort=False)]  # qid, pid, relevancy

        assert passages_per_query >= 5
        self.passage_per_q = passages_per_query
        self.p_tensors = passages_tensors
        self.q_tensors = queries_tensors

        if fixed_samples:
            self.select_p_idx = {}

    def __len__(self):
        return len(self.valid_q_indexes)

    def _set_generator(self):

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int32).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        return generator

    def __getitem__(self, q_idx: int):
        q_idx = self.valid_q_indexes[q_idx]

        passage_collection = self.all_dataframe[q_idx]
        qid = passage_collection.qid.iloc[0]

        num_positives = len(passage_collection[passage_collection.relevancy == 1])
        assert num_positives > 0
        num_passages = len(passage_collection)
        assert num_passages >= self.passage_per_q

        generator = self._set_generator()

        if hasattr(self, 'select_p_idx'):
            if num_positives in self.select_p_idx:
                p_index = self.select_p_idx[num_positives]
            else:
                p_index = self._get_p_idx(generator, num_passages, num_positives)
                self.select_p_idx[num_positives] = p_index
        else:
            p_index = self._get_p_idx(generator, num_passages, num_positives)

        pids = passage_collection.loc[p_index, 'pid'].values.reshape(-1).tolist()

        y = passage_collection.loc[:self.passage_per_q - 1, ['relevancy']].values.reshape(-1)  # (N,)

        if self.return_tensors == 'tuple':
            if self.fake_tensor:
                x = [torch.rand(1, 300), torch.rand(self.passage_per_q, 300)]
            else:
                query = self.q_tensors[qid].reshape(-1, 300)
                passages = torch.concatenate([self.p_tensors[pid].reshape(-1, 300) for pid in pids])
                x = [query, passages]

            if self.shuffle_passages:
                shuffle_idx = torch.randperm(self.passage_per_q)
                y = y[shuffle_idx]
                x[1] = x[1][shuffle_idx, ...]
            return [x, torch.from_numpy(y)]
        elif self.return_tensors == 'cat':
            if self.fake_tensor:
                x = torch.rand(self.passage_per_q, 2, 300)
            else:
                query = self.q_tensors[qid].reshape(-1).repeat(self.passage_per_q, 1)
                passages = torch.concatenate([self.p_tensors[pid].reshape(-1, 300) for pid in pids])
                x = torch.stack([query, passages], dim=1)  # (N, 2, 300)

            if self.shuffle_passages:
                shuffle_idx = torch.randperm(self.passage_per_q)
                y = y[shuffle_idx]
                x = x[shuffle_idx, ...]
            return [x, torch.from_numpy(y)]
        else:
            raise ValueError

    # def _get_p_idx(self, generator, num_passages, num_positives):
    #     p_index = torch.arange(num_positives).tolist()
    #     p_index += (torch.randperm(num_passages - num_positives, generator=generator) + num_positives).tolist()[
    #                :self.passage_per_q - num_positives]
    #     return p_index

    def _get_p_idx(self, generator, num_passages, num_positives):
        p_index = torch.cat((torch.arange(num_positives),
                             torch.randperm(num_passages - num_positives, generator=generator)[
                             :self.passage_per_q - num_positives] + num_positives))
        return p_index.tolist()


class ValidationDataset(Dataset):

    def __init__(self, all_dataframe: pd.DataFrame,
                 train_p_tensors: dict[int, torch.Tensor],
                 val_p_tensors: dict[int, torch.Tensor],
                 queries_tensors: dict[int, torch.Tensor],
                 return_tensors, fake_tensor=True):
        super(Dataset).__init__()
        self.val_p_tensors = val_p_tensors
        self.fake_tensor = fake_tensor
        self.return_tensors = return_tensors
        # qid, pid, relevancy
        self.all_dataframe = {key: [value.reset_index(drop=True, inplace=True), value][1] for key, value in
                              all_dataframe.groupby('q_idx', sort=False)}

        self.p_tensors = train_p_tensors
        self.q_tensors = queries_tensors

    def __len__(self):
        return len(self.all_dataframe)

    def get_p_tensor(self, pid):
        try:
            return self.p_tensors[pid]
        except KeyError:
            return self.val_p_tensors[pid]

    def __getitem__(self, q_idx: int):

        passage_collection = self.all_dataframe[q_idx]

        qid = passage_collection.qid.iloc[0]
        pids = passage_collection.loc[:, 'pid'].values.reshape(-1).tolist()

        result = []
        if type(self.return_tensors) is str:
            passagess = torch.concatenate([self.get_p_tensor(pid).reshape(-1, 300) for pid in pids])
            if self.return_tensors == 'tuple':

                if self.fake_tensor:
                    x = (torch.rand(1, 300), torch.rand(len(pids), 300))
                else:
                    query = self.q_tensors[qid].reshape(-1, 300)
                    x = (query, passagess)

            elif self.return_tensors == 'cat':
                if self.fake_tensor:
                    x = torch.rand(len(pids), 2, 300)
                else:
                    query = self.q_tensors[qid].reshape(-1).repeat(len(pids), 1)
                    x = torch.stack([query, passagess], dim=1)  # (N, 2, 300)

            y = passage_collection.loc[:, ['relevancy']].values.reshape(-1)  # (N,)
            result += [x, torch.from_numpy(y)]

        return result

    # def evaluate(self, q_idx: int):


if __name__ == '__main__':
    from utils import train_raw_df, queries_embeddings, map_location, load_passages_tensors
    from torch.utils.data import DataLoader

    df = pd.read_parquet(train_raw_df)

    # q_tensors = torch.load(queries_embeddings, map_location=map_location)
    # p_tensors = load_passages_tensors(first=1)
    q_tensors = None
    p_tensors = None
    dataset = CustomDataset(df,
                            queries_tensors=q_tensors,
                            fake_tensor=True,
                            passages_per_query=5,
                            passages_tensors=p_tensors,
                            fixed_samples=True,
                            return_tensors='tuple',

                            generator=None)
    dataloader = DataLoader(dataset, batch_size=10)
    for i, batch in enumerate(dataloader):
        print(batch)
        if i > 5: break
