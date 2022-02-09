# How to reconstruct ICU files

## Getting set up

Follow the installation instructions for standard IceCube reco (in `freeDOM/examples/i3module`). The process is the same here.  

## Reco'ing files

Generate a list(s) of i3 files to reconstruct (again, see instructions in `freeDOM/examples/i3module` for more details). Then, to launch the reconstruction processes, use the `run_ICU_reco.py` script. Run `python run_ICU_reco.py --help` for more information about the supported command line parameters. For example, one can process a small portion of all the i3 files specified in a text file called `test_list.txt` with the following command:
```sh
python run_ICU_reco.py --in_list test_list.txt --outdir test_out --cuda_device 0 --n_frames 11
```

To process an entire i3 file, simply omit the `--n_frames` argument.