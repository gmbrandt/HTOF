from astropy.time import Time


def gaia_obmt_to_tcb_julian_year(obmt):
    """
    convert OBMT (on board mission timeline) to TCB julian years via
    https://gea.esac.esa.int/archive/documentation/GDR2/Introduction/chap_cu0int/cu0int_sec_release_framework/cu0int_ssec_time_coverage.html
    Equation 1.1
    :param obmt: on-board mission timeline in units of six-hour revolutions since launch. OBMT (in revolutions)
    :return: astropy.time.Time
    """
    tcbjy = 2015 + (obmt - 1717.6256)/(1461)
    return Time(tcbjy, scale='tcb', format='jyear')
