"""
Set a result directory in the source.
Extracts sfa2 modules and whiteners from all results files (pickled :py:class:`core.result.Result` objects)
and pickles them to files in the directory.

"""

import pickle

PATH = "/local/results/reorder_o18c/"

if __name__ == "__main__":

    for i in range(31):
        with open(PATH+"res{}.p".format(i), 'rb') as f:
            res = pickle.load(f)
        res.sfa2S.save(PATH+"res{}_sfa2S.sfa".format(i))
        res.sfa2E.save(PATH + "res{}_sfa2E.sfa".format(i))
        with open(PATH+"res{}_whitener.p".format(i), 'wb') as w:
            pickle.dump(res.whitener, w)
