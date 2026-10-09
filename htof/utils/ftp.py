from ftplib import FTP_TLS, error_perm
import warnings
import os


def download_and_save_hip21_data_to(star_id, outpath):
    if int(star_id) < 1 or int(star_id) > 120404:
        raise RuntimeError("Can not download data. The Hipparcos star ID is likely invalid.")
    file_name = 'H' + str(star_id).zfill(6) + '.d'
    subdir = file_name[:4]
    # the filepath on the anonymous@ftp.cosmos.esa.int ftp server. I think this path needs to be in the
    # unix / forward slashes, so we are not using os.path here because on windows it will use \ slashes.
    fullpath = f'HIPPARCOS_PUBLIC_DATA/ResRec_JavaTool_2014/{subdir}/{file_name}'

    HOSTNAME = "ftp.cosmos.esa.int"
    USERNAME = "anonymous"
    PASSWORD = "https://github.com/gmbrandt/HTOF" # so that the host knows where this request is coming from.
    # Log into the ESA FTP server.
    try:
        ftps = FTP_TLS(HOSTNAME, timeout=30)
        ftps.login(USERNAME, PASSWORD)
        ftps.prot_p()
    except Exception:
        warnings.warn("Connecting to European Space Agency FTP failed.")
        raise RuntimeError("Connecting to European Space Agency FTP failed. Try again later, or download this"
                            " file manually.")
    # Download the IAD file.
    try:
        # Write file in binary mode
        with open(outpath, "wb") as file:
            # Command for Downloading the file "RETR filename"
            ftps.retrbinary(f"RETR {fullpath}", file.write)
    except Exception:
        os.remove(outpath)
        raise RuntimeError("Downloading the IAD file failed. Try again later, or download this"
                            " file manually. Also check if the star_id is a valid Hipparcos ID.")
    # log out of the FTP server.
    ftps.quit()
    return None

