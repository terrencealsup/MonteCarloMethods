Terrence Alsup

April 17, 2019
Monte Carlo Methods HW 4


---------------------------------------------------------------------------
The following four Python files are included:

XYmodel.py
ex60_alsup.py
ex61_alsup.py
ex62_alsup.py

The last 3 produce the plots and IACs for exercises 60, 61, and 62 respectively.

To run and compile any of the files simply go to command line in the directory containing the files and type:

python FILE_NAME.py

where FILE_NAME is replaced appropriately.  In each of the last 3 files there are functions called "test_IAC" and
"test_sampler".  "test_sampler" computes the plots shown in the pdf with the vectors on the unit circle.  "test_IAC"
computes the IAC times given different parameters, which can be altered at the beginning of the method.  See the
documentation in the file for more information.

---------------------------------------------------------------------------
Dependencies

The code uses Anaconda 3, which can be loaded on a CIMS desktop by calling

module load anaconda3

These files have the following Python package dependencies:

numpy==1.15.4
matplotlib==3.0.3
acor==1.1.1



