import unittest
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import pandas

from tests import thisPath, resultsP863File
from tests.data import dataPath
from degradeSpecSub import applySpecSub

class SpecSubDegradeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputPath = thisPath / 'output'
        for wvFile in cls.outputPath.glob('*.wav'):
            wvFile.unlink()
        cls.outputPath.mkdir(exist_ok=True)
        cls.testFile = dataPath / Path('German_P835_16_sentences_4convergence.wav')

    def test_specsub(self):
        # storage for generated files
        df = pandas.DataFrame(columns=['Filename', 'SNR', 'OSF', 'TimeConst', 'PowExp']).set_index('Filename')

        # load signal
        fs = librosa.get_samplerate(self.testFile)
        s, _ = librosa.load(self.testFile, fs)

        for snr in [10, 0, -10, -20, -30]: # 20, 10, 5,
            for osf in [0.5, 1.0, 1.5, 2.0]:
                for tc in [0.005, 0.035, 0.125, 0.250]:
                    for pow_exp in [0.5, 1.0, np.sqrt(2), 2.0]:
                        with self.subTest(snr=snr, osf=osf, tc=tc, pow_exp=pow_exp):
                            # run with settings
                            outputFile = self.outputPath / Path('processed_snr=%d_osf=%.2f_tc=%d_pe=%.2f.wav' % (snr, osf, tc*1000, pow_exp))
                            d = applySpecSub(s, fs, snr=snr, osf=osf, tcNoise=tc, tcSpeech=tc)

                            signal = np.vstack((d, s)).T
                            sf.write(outputFile, signal, fs)
                            df.loc[str(outputFile), :] = [snr, osf, tc, pow_exp]
                            df.to_excel(resultsP863File)

if __name__ == '__main__':
    unittest.main()
