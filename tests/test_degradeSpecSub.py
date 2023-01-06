import unittest
import os
from typing import List
from pathlib import Path
from resampy import resample
import numpy as np
import soundfile as sf
import pandas
from concurrent.futures import ProcessPoolExecutor

from tests import thisPath, resultsP863File, resultColumns, resultIndices, resultIdxRange
from tests.data import downloadETSITestFile, TestFilesETSI
from degradeSpecSub import applySpecSub
from p56.asl import calculateP56ASLEx
from helper import FS

class SpecSubDegradeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputPath = thisPath / 'output'
        #for wvFile in cls.outputPath.glob('*.wav'):
        #    wvFile.unlink()
        cls.outputPath.mkdir(exist_ok=True)
        cls.testFile = None

    @staticmethod
    def _process_sequence(s: np.ndarray, fs: int, outputFile: Path,
                          snr: float, osf: float, tc: float, pow_exp: float,
                          targetAsl: float = -26.0) -> pandas.DataFrame:

        d = applySpecSub(s, fs, targetAsl, snr=snr, osf=osf, tcNoise=tc, tcSpeech=tc, pow_exp=pow_exp)

        # rescale to -26 dBov
        asl, _ = calculateP56ASLEx(d, fs, preFilter='FB')
        d *= np.power(10, (targetAsl - asl) / 20)

        # store degraded and reference in one file
        signal = np.vstack((d, s)).T

        # use 16-bit (neeed for POLQA testing)
        sf.write(outputFile, signal, fs, subtype='PCM_16', format='FLAC')


    @staticmethod
    def _process_sequences(testFiles: List[Path], outputPath: Path, fs: int=FS, maxWorkers: int = os.cpu_count()-1) -> pandas.DataFrame:
        # storage for generated files
        if resultsP863File.is_file():
            df = pandas.read_excel(resultsP863File, index_col=resultIdxRange)
        else:
            df = pandas.DataFrame(columns=resultColumns+resultIndices).set_index(resultIndices)

        # generate all samples via multiprocessing
        with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            # start tasks
            results = dict()
            for testFile in testFiles:
                # load & resample signal
                s, fs1 = sf.read(testFile)
                if fs1 != fs:
                    s = resample(s, fs1, fs)

                # iterate over internal pseudo-noise-reduction parameters:
                for snr in [10, 5, 2.5, 0, -2.5, -5, -10, -20, -30]:  # SNR between speech and speech-shaped noise
                    for osf in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.90, 1.0]:  # over-subtraction factor
                        for tc in [0.005, 0.035, 0.125, 0.250]:  # time constant for smoothing
                            for pow_exp in [0.5, 1.0, np.sqrt(2), 2.0]:  # power exponent for Wiener gain

                                # run with given settings
                                outputFile = outputPath / Path('processed_%s_snr=%d_osf=%.2f_tc=%d_pe=%.2f.flac' % (
                                    testFile.stem, snr, osf, tc * 1000, pow_exp))

                                if not outputFile.is_file():
                                    results[outputFile] = executor.submit(SpecSubDegradeTestCase._process_sequence,
                                                                          s, fs, outputFile, snr, osf, tc, pow_exp)

                                # store information for P.863 calculation in other unit test
                                key = str(outputFile)
                                if key not in df.index:
                                    df.loc[key, :] = [testFile.stem, snr, osf, tc, pow_exp, -1.0]

            # wait for tasks
            nbrItems = df.shape[0]
            print(f"Waiting for {len(results)}/{nbrItems} items to complete...")
            for i, (key, futureResult) in enumerate(results.items()):
                e = futureResult.exception()
                if e is None:
                    dfProc = futureResult.result()

                    # store in output
                    if dfProc is not None:
                        df = df.combine_first(dfProc)
                        df.to_excel(resultsP863File)
                else:
                    print(str(e))

        # final check: remove all rows where the file does not exist
        remove = []
        for key, row in df.iterrows():
            if not Path(key).is_file():
                remove.append(key)

        df = df.drop(labels=remove)

        df.to_excel(resultsP863File)

    def test_specsub(self):


        # run for all test signals
        testFiles = []
        for tstFile in TestFilesETSI:
            with self.subTest(testFile=tstFile.value):
                testFile = downloadETSITestFile(tstFile, proxy="http://127.0.0.1:3128")
                testFiles.append(testFile)

        self._process_sequences(testFiles, outputPath=self.outputPath)

if __name__ == '__main__':
    unittest.main()
