#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/HeaDef/lammpsRuns/NiNatom100KTwin11th

MEAM_library_DIR=/home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials

module load openmpi/4.0.2-gnu730

python3 twinBoundaries.py  /home/kamran.karimi1/Project/git/HeaDef/lammpsRuns/../postprocess 3.52 35.0 20.0 20.0 data.txt
mpirun --oversubscribe -np 4 $EXEC_DIR/lmp_mpi < in.minimization -echo screen -var OUT_PATH . -var PathEam /home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials -var INC /home/kamran.karimi1/Project/git/HeaDef/lammpsRuns/lmpScripts  -var buff 0.0 -var buffy 5.0 -var nevery 1000 -var ParseData 1 -var DataFile data.txt -var DumpFile dumpMin.xyz -var WriteData data_minimized.txt
mpirun --oversubscribe -np 4 $EXEC_DIR/lmp_mpi < in.thermalizeNVT -echo screen -var OUT_PATH . -var PathEam /home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials -var INC /home/kamran.karimi1/Project/git/HeaDef/lammpsRuns/lmpScripts  -var buff 0.0 -var buffy 5.0 -var T 5.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_5K.dat
mpirun --oversubscribe -np 4 $EXEC_DIR/lmp_mpi < in.shearDispTemp -echo screen -var OUT_PATH . -var PathEam /home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials -var INC /home/kamran.karimi1/Project/git/HeaDef/lammpsRuns/lmpScripts  -var buff 0.0  -var buffy 5.0 -var T 5.0 -var GammaXY 0.1 -var GammaDot 1.0e-05 -var ndump 100 -var ParseData 1 -var DataFile Equilibrated_5K.dat -var DumpFile dumpSheared.xyz
