import unittest
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import pandas

from tests import thisPath, resultsP863File
from tests.data import downloadETSITestFile, TestFilesETSI
from degradeSpecSub import applySpecSub

class SpecSubDegradeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputPath = thisPath / 'output'
        #for wvFile in cls.outputPath.glob('*.wav'):
        #    wvFile.unlink()
        cls.outputPath.mkdir(exist_ok=True)
        cls.testFile = None

    def test_specsub(self):
        # storage for generated files
        if resultsP863File.is_file():
            df = pandas.read_excel(resultsP863File, index_col='Filename')
        else:
            df = pandas.DataFrame(columns=['Filename', 'Language', 'SNR', 'OSF', 'TimeConst', 'PowExp']).set_index('Filename')

        # run for all test signals
        for tstFile in TestFilesETSI:
            with self.subTest(testFile=tstFile.value):
                self.testFile = downloadETSITestFile(tstFile, proxy="https://127.0.0.1:3128")

                # load signal
                fs = librosa.get_samplerate(self.testFile)
                s, _ = librosa.load(self.testFile, fs)

                # iterate over internal pseudo-noise-reduction parameters:
                for snr in [10, 0, -10, -20, -30]: # SNR between speech and speech-shaped noise
                    for osf in [0.5, 1.0, 1.5, 2.0]: # over-subtraction factor
                        for tc in [0.005, 0.035, 0.125, 0.250]: # time constant for smoothing
                            for pow_exp in [0.5, 1.0, np.sqrt(2), 2.0]: # power exponent for Wiener gain
                                with self.subTest(snr=snr, osf=osf, tc=tc, pow_exp=pow_exp):
                                    # run with settings
                                    outputFile = self.outputPath / Path('processed_%s_snr=%d_osf=%.2f_tc=%d_pe=%.2f.wav' % (tstFile.value, snr, osf, tc*1000, pow_exp))
                                    if not outputFile.is_file():
                                        d = applySpecSub(s, fs, snr=snr, osf=osf, tcNoise=tc, tcSpeech=tc)

                                        # store degraded and reference in one file
                                        signal = np.vstack((d, s)).T
                                        sf.write(outputFile, signal, fs)

                                    # store information for P.863 calculation in other unit test
                                    df.loc[str(outputFile), :] = [tstFile.value, snr, osf, tc, pow_exp]
                                    df.to_excel(resultsP863File)

if __name__ == '__main__':
    unittest.main()
