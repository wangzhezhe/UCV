#!/bin/bash
#BSUB -P csc143
#BSUB -W 01:59
#BSUB -nnodes 1

#BSUB -J run_beetle_serial
#BSUB -o run_beetle_serial.%J.out
#BSUB -e run_beetle_serial.%J.err 

CURRDIR=$(pwd)
LOGDIRNAME=run_beetle_serial_log

mkdir -p $CURRDIR/$LOGDIRNAME

cd $MEMBERWORK/csc143

rm -rf $LOGDIRNAME
mkdir $LOGDIRNAME

cd $LOGDIRNAME

ln -s $CURRDIR/../../install_scripts/summit_gpu/install/UCV/ucv_reduce_umc ucv_reduce_umc
ln -s $CURRDIR/../../install_scripts/summit_gpu/install/UCV/uvm_point_neighborhood uvm_point_neighborhood

DATANAME=beetle_496_832_832.vtk
DATASETPATH=/gpfs/alpine/proj-shared/csc143/zhewang/datasets/uncertainty/$DATANAME
FIELD=ground_truth

export OMP_NUM_THREADS=1

#block size compare
jsrun -n1 -a1 -c1 -g1 ./uvm_point_neighborhood --vtkm-device serial $DATASETPATH $FIELD uni 4 900 1000 &> point_neighborhood_uni_1.log

jsrun -n1 -a1 -c1 -g1 ./uvm_point_neighborhood --vtkm-device serial $DATASETPATH $FIELD uni 4 900 1000 &> point_neighborhood_uni_2.log

jsrun -n1 -a1 -c1 -g1 ./uvm_point_neighborhood --vtkm-device serial $DATASETPATH $FIELD ig 4 900 1000 &> point_neighborhood_ig.log

jsrun -n1 -a1 -c1 -g1 ./uvm_point_neighborhood --vtkm-device serial $DATASETPATH $FIELD mg 4 900 1000 &> point_neighborhood_mg.log


#there are issues if we set g as 0 even if for the openmp backend
jsrun -n1 -a1 -c1 -g1 ./ucv_reduce_umc --vtkm-device serial $DATASETPATH $FIELD uni 4 900 1000 &> ucv_umc_serial_uni_1.log

jsrun -n1 -a1 -c1 -g1 ./ucv_reduce_umc --vtkm-device serial $DATASETPATH $FIELD uni 4 900 1000 &> ucv_umc_serial_uni_2.log

jsrun -n1 -a1 -c1 -g1 ./ucv_reduce_umc --vtkm-device serial $DATASETPATH $FIELD ig 4 900 1000 &> ucv_umc_serial_ig.log

jsrun -n1 -a1 -c1 -g1 ./ucv_reduce_umc --vtkm-device serial $DATASETPATH $FIELD mg 4 900 1000 &> ucv_umc_serial_mg_1k.log

jsrun -n1 -a1 -c1 -g1 ./ucv_reduce_umc --vtkm-device serial $DATASETPATH $FIELD mg 4 900 2000 &> ucv_umc_serial_mg_2k.log

cp *.log $CURRDIR/$LOGDIRNAME

# clean the run dir
#rm -r $LOGDIRNAME