import pandas
import pyarrow.parquet as pq
import os
import subprocess

ROOT_PATH = "/home/kollasfa/MasterThesis/DATA/BTS"


def clean_irsa_result():
    df = pandas.read_csv(os.path.join(ROOT_PATH, "irsa_results.csv"))
    clean_frame = df.sort_values('ngoodobs', ascending=False).drop_duplicates('ztf_01').sort_values("cntr_01")
    with open(os.path.join(ROOT_PATH, "bts_meta.csv", "w")) as clean_path:
        clean_frame.to_csv(clean_path)


def download_parquets():
    df = pandas.read_csv(os.path.join(ROOT_PATH, "bts_meta.csv"))

    def runcmd(cmd, verbose=False, *args, **kwargs):

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        std_out, std_err = process.communicate()
        if verbose:
            print(std_out.strip(), std_err)
        pass

    for idx, row in df.iterrows():
        field_id = str(row['field']).zfill(6)
        cmd = 'wget -r -np -nH -R "index.html*" https://irsa.ipac.caltech.edu/data/ZTF/lc/lc_dr11/0/field{0}/'.format(field_id)
        print("Getting data from field {0}".format(field_id))
        runcmd(cmd)


if __name__ == '__main__':
    # clean_irsa_result()
    download_parquets()