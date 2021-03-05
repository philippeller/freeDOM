# How to reconstruct i3 files, a recipe

## Getting set up

1. In an Icetray environement, install freeDOM light version, e.g:
```
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh
cd freedom
pip install --user --editable .
```


2. Exit icetray and start fresh (otherwise env variables stick around).
Now in your python installation where you want to run tensorflow, do a full install

```
pip install --editable .[full]
```

obtain the model files from `cobalt:/data/user/peller/freeDOM/resources`
adjust paths in `freeDOM/examples/i3module/service_control.py`

## Reco'ing files

Generate a list(s) of files to be reco'ed, typically with 50-100 i3 file paths each, for example with the script `freeDOM/examples/i3module/make_file_lists.py`
```
python make_file_lists.py -n 100 -p file_list "/some_path/oscNext/level7_v01.04/120000/*.i3.zst"
```

Launch the reconstruction of i3 paths in a given file list by (the first argument, here `0`, is the GPU number):
```
./run_i3_reco.sh 0 file_list_0.txt /path_to_outdir
```
Go drink a coffee or two and wait...

(For testing, or processing partial files, you can specify `--n_frames N`, where N could be 13, as an option to `i3_reco.py`, e.g. in `run_i3_reco.sh`)
