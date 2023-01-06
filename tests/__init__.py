# -*- coding: utf-8 -*-
"""
Created on Apr 14 2021 13:07

@author: Jan.Reimes
"""

from pathlib import Path
import numpy as np


thisPath = Path(__file__).parent
resultsP863File = thisPath / Path('Results-P863.xlsx')

resultColumns = ['SourceFile', 'SNR', 'OSF', 'TimeConst', 'PowExp', 'MOS-LQO']
resultIndices = ['Filename']
resultIdxRange = np.arange(len(resultIndices)).tolist()


if __name__ == "__main__":
    pass
