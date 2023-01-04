import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas

from tests import resultsP863File, resultIdxRange
from p863 import runPOLQA

class P863CalcTestCase(unittest.TestCase):
    def test_calcP863(self):
        if not resultsP863File.is_file():
            self.assertFalse(True, 'Cannot find %s - please run test for generation of files first' % (resultsP863File.name))

        # analysis parameters for the ETSI test files:
        duration = 8.0
        startTime = 16.0
        nbrRanges = 8

        # init storage and new column
        df = pandas.read_excel(resultsP863File, index_col=resultIdxRange)
        if not ('MOS-LQO' in df.columns):
            df['MOS-LQO'] = pandas.NA

        # calculate POLQA scores
        for i, (key, row) in enumerate(df.iterrows()):
            print("[%d/%d] %s" % (i+1, df.shape[0], Path(key).name))
            if pandas.isna(df.loc[key, 'MOS-LQO']) or (df.loc[key, 'MOS-LQO'] < 1.0):
                dfPerFile = None
                for i in range(nbrRanges):
                    res, _ = runPOLQA(key, key, chNbrRef=2, chNbrDeg=1,
                                   timeRangeStart=startTime + i*duration,
                                   timeRangeDuration=duration)
                    if res.shape[0] > 0:
                        if dfPerFile is None:
                            dfPerFile = pandas.DataFrame(columns=res.index)

                        dfPerFile.loc[i, :] = res

                # store in output
                if not dfPerFile is None:
                    df.loc[key, 'MOS-LQO'] = dfPerFile['MOS-LQO'].mean()
                    df.to_excel(resultsP863File)

    def test_analyse_P863_results(self):
        # try to automatically select the four best noise reduction parameters that generate:
        # - equidistant MOS-LQO for anchoring (~1.0 / ~2.0 / ~3.0 / ~4.0 - 5.0/max is given by direct reference)
        # - most consistent results across language/source files

        TargetMOS = [1.0, 2.0, 3.0, 4.0]

        if resultsP863File.is_file():
            # load
            df = pandas.read_excel(resultsP863File, index_col=resultIdxRange)
            df = df.dropna()

            # average across samples and languages
            df = df.groupby(['SNR','OSF','TimeConst','PowExp']).agg({'MOS-LQO': ['mean', 'min', 'max', 'std']})

            # add delta column
            df[('MOS-LQO', 'delta')] = df[('MOS-LQO', 'max')] - df[('MOS-LQO', 'min')]

            # make histogram
            edges = np.arange(5) + 0.5 # "target MOS" in

            fig, ax = plt.subplots(1,1)
            ax.hist(df[('MOS-LQO', 'mean')], bins=edges)
            ax.set_xlabel('Avg. MOS-LQO')
            ax.set_ylabel('Count')
            ax.set_xlim([1, 5])
            ax.grid(True)
            fig.tight_layout()

            # evaluation criteria:
            # - delta and std (=multiplication) to be minimized -> score
            # - as close as possible to centre of bin
            df['score'] = df[('MOS-LQO', 'delta')] * df[('MOS-LQO', 'std')]

            # iterate over edges
            for i in range(len(edges)-1):
                # filter items
                lower = edges[i]
                upper = edges[i+1]
                centre = 0.5 * (upper + lower)
                dfBin = df[(df[('MOS-LQO', 'mean')] >= lower) & (df[('MOS-LQO', 'mean')] < upper)]

                # apply difference to centre
                dfBin['score'] *= np.abs(df[('MOS-LQO', 'mean')] - centre)

                # sort and pick first
                dfBin = dfBin.sort_values('score', ascending=True)
                print(dfBin.iloc[0]['MOS-LQO'])

            plt.show()


if __name__ == '__main__':
    unittest.main()
