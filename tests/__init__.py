# -*- coding: utf-8 -*-
"""
Created on Apr 14 2021 13:07

@author: Jan.Reimes
"""

from pathlib import Path
import numpy as np

thisPath = Path(__file__).parent
resultsP863File = thisPath / Path('Results-P863.xlsx')

resultColumns = ['Language', 'SNR', 'OSF', 'TimeConst', 'PowExp']
resultIndices = ['Filename']
resultIdxRange = np.arange(len(resultIndices))

if __name__ == "__main__":
    pass
