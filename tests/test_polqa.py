import unittest
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
