import pandas as pd
import numpy as np
from htof.utils.data_utils import merge_consortia


def test_merge_consortia():
    data = pd.DataFrame([[133, 'F', -0.9053, -0.4248,  0.6270,  1.1264,  0.5285, -2.50,  2.21,  0.393],
                         [133, 'N', -0.9051, -0.4252,  0.6263, 1.1265,  0.5291, -1.18, 1.59, 0.393]],
                        columns=['A1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10'])
    merged_orbit = merge_consortia(data)
    import pdb
    pdb.set_trace()
    assert np.isclose(merged_orbit['IA9'], 1.498373)  # merged error
    assert np.isclose(merged_orbit['IA8'], -1.505620)  # merged residual


def test_merge_consortia_equal_on_flipped_rows():
    data = pd.DataFrame([[133, 'F', -0.9053, -0.4248,  0.6270,  1.1264,  0.5285, -2.50,  2.21,  0.393],
                         [133, 'N', -0.9051, -0.4252,  0.6263, 1.1265,  0.5291, -1.18, 1.59, 0.393]],
                        columns=['A1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10'])
    merged_orbit1 = merge_consortia(data)
    data = pd.DataFrame([[133, 'N', -0.9051, -0.4252,  0.6263, 1.1265,  0.5291, -1.18, 1.59, 0.393],
                         [133, 'F', -0.9053, -0.4248,  0.6270,  1.1264,  0.5285, -2.50,  2.21,  0.393]],
                        columns=['A1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10'])
    assert merged_orbit1.equals(merge_consortia(data))