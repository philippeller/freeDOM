"""Generate input file lists to use with run_i3_reco.sh """
import glob
from argparse import ArgumentParser


def make_file_list(ofilename, files):
    with open(ofilename, "w") as out_f:
        for file in files:
            out_f.write(f"{file}\n")


def make_file_lists(n_per_list, prefix, files):
    files = glob.glob(files)
    file_splits = (files[i : i + n_per_list] for i in range(0, len(files), n_per_list))
    for i, file_split in enumerate(file_splits):
        make_file_list(f"{prefix}_{i}.txt", file_split)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--n_per_list", "-n", type=int, default=50, help="""n files per file list"""
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="test_list",
        help="""output file name prefix""",
    )
    parser.add_argument("files", help="""Files to split into file lists""")

    args = parser.parse_args()

    make_file_lists(args.n_per_list, args.prefix, args.files)


if __name__ == "__main__":
    main()
