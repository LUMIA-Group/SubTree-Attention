import torch


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 5
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {ind+1}')
            print(f'Final Train: {result[ind, 0]:.2f}')
            print(f'Final Test: {result[ind, 2]:.2f}')
            self.test=result[ind, 2]
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test=r.mean()
            return
    
    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')