import pandas as pd
import numpy as np
from htof.utils.data_utils import merge_consortia, safe_concatenate


def test_merge_consortia():
    data = pd.DataFrame([[133, 'F', -0.9053, -0.4248,  0.6270,  1.1264,  0.5285, -2.50,  2.21,  0.393],
                         [133, 'N', -0.9051, -0.4252,  0.6263, 1.1265,  0.5291, -1.18, 1.59, 0.393],
                         [271, 'N', -0.1051, -0.1252, 0.1263, 0.1265, 0.1291, -0.18, 0.59, 0.193]],
                        columns=['A1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10'])
    merged_orbit = merge_consortia(data)
    assert len(merged_orbit) == 2
    assert np.isclose(merged_orbit['IA9'].iloc[0], 1.498373)  # merged error
    assert np.isclose(merged_orbit['IA8'].iloc[0], -1.505620)  # merged residual
    assert np.isclose(merged_orbit['IA8'].iloc[1], -0.18)  # single orbit un-touched residual


def test_merge_consortia_equal_on_flipped_rows():
    data1 = pd.DataFrame([[133, 'F', -0.9053, -0.4248,  0.6270,  1.1264,  0.5285, -2.50,  2.21,  0.393],
                         [133, 'N', -0.9051, -0.4252,  0.6263, 1.1265,  0.5291, -1.18, 1.59, 0.393]],
                        columns=['A1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10'])
    data2 = pd.DataFrame([[133, 'N', -0.9051, -0.4252,  0.6263, 1.1265,  0.5291, -1.18, 1.59, 0.393],
                         [133, 'F', -0.9053, -0.4248,  0.6270,  1.1264,  0.5285, -2.50,  2.21,  0.393]],
                        columns=['A1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10'])
    pd.testing.assert_frame_equal(merge_consortia(data2), merge_consortia(data1), check_less_precise=2)


def test_safe_concatenate():
    a, b = np.arange(3), np.arange(3, 6)
    assert np.allclose(a, safe_concatenate(a, None))
    assert np.allclose(b, safe_concatenate(None, b))
    assert None is safe_concatenate(None, None)
    assert np.allclose(np.arange(6), safe_concatenate(a, b))
