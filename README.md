# cdmt
CDMT, for *Coherent Dispersion Measure Trials*, is a software program to perform coherent dedispersion on complex voltage data from the LOFAR telescope and to coherently dedisperse to many different dispersion measure trials. It reads HDF5 input data and generates [SIGPROC](http://sigproc.sourceforge.net/) compatible filterbank files.

The software uses NVIDIA GPUs to accelerate the computations and hence requires compilation with `nvcc` against the `cufft` and `hdf5` libraries.

Presently the code is only capable of reading LOFAR HDF5 complex voltage data. If you want to use `cdmt` with a different input data type, let me know.

## Fork details
This code is currently being modified to process MWA raw voltages in VDIF format, as output by [VCSBeam](https://github.com/CIRA-Pulsars-and-Transients-Group/vcsbeam).
Some parts of the code have been taken from [David McKenna's fork](https://github.com/David-McKenna/cdmt) of the repository.