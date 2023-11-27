#!/bin/bash
set -e

module load cuda
module load gcc
module load cmake
module load eigen/3.3.9

build_jobs=4
mkdir -p summit_gpu
cd summit_gpu

HERE=`pwd`
source $HERE/../settings.sh

SOFTWARE_SRC_DIR="$HERE/src"
SOFTWARE_BUILD_DIR="$HERE/build"
SOFTWARE_INSTALL_DIR="$HERE/install"

mkdir -p $SOFTWARE_SRC_DIR
mkdir -p $SOFTWARE_BUILD_DIR
mkdir -p $SOFTWARE_INSTALL_DIR


echo "====> Installing kokkos"

kokkos_src_dir="$SOFTWARE_SRC_DIR/kokkos-$KOKKOS_VERSION"
kokkos_build_dir="$SOFTWARE_BUILD_DIR/kokkos-$KOKKOS_VERSION"
# leting build dir == install dir
# there are issues if setting install dir with the version info
kokkos_install_dir="$SOFTWARE_INSTALL_DIR/kokkos-$KOKKOS_VERSION"

if [ -d $kokkos_install_dir ]; then
    echo "====> skip, $kokkos_install_dir already exists," \
             "please remove it if you want to reinstall it"
else

    rm -rf ${kokkos_src_dir}
    git clone -b master $KOKKOS_REPO ${kokkos_src_dir}
    cd ${kokkos_src_dir}
    git checkout 3.7.01
 
    # switch the device in the cpp file
    CXX=${kokkos_src_dir}/bin/nvcc_wrapper
    sed -i 's/sm_35/sm_70/g' $CXX
    # refer to 
    # https://gitlab.kitware.com/vtk/vtk-m/-/blob/master/.gitlab/ci/docker/ubuntu1804/kokkos-cuda/Dockerfile
    cmake -S ${kokkos_src_dir} -B ${kokkos_build_dir} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${kokkos_install_dir} \
    -DBUILD_SHARED_LIBS=ON\
    -DCMAKE_CXX_FLAGS=-fPIC \
    -DCMAKE_CXX_STANDARD=14 \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_CUDA_LDG_INTRINSIC=ON \
    -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
    -DKokkos_ENABLE_CUDA_UVM=ON \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc

    cmake --build ${kokkos_build_dir} -j${build_jobs}
    
    cmake --install ${kokkos_build_dir}

    # there are some issues for installing by other ways
    # make install 

fi


echo "====> Installing vtk-m"
VTKM_SRC_DIR="$SOFTWARE_SRC_DIR/vtk-m"
VTKM_BUILD_DIR="$SOFTWARE_BUILD_DIR/vtk-m"
VTKM_INSTALL_DIR="$SOFTWARE_INSTALL_DIR/vtk-m"

# check the install dir
if [ -d $VTKM_INSTALL_DIR ]; then
    echo "====> skip, $VTKM_INSTALL_DIR already exists," \
             "please remove it if you want to reinstall it"
else
    echo $VTKM_SRC_DIR
    echo $VTKM_BUILD_DIR
    echo $VTKM_INSTALL_DIR
    # check vktm source dir
    if [ ! -d $VTKM_SRC_DIR ]; then
    # clone the source
    cd $SOFTWARE_SRC_DIR
    git clone $VTKM_REPO
    cd $VTKM_SRC_DIR
    git checkout $VTKM_VERSION
    fi
    
    cd $HERE

    # build and install
    echo "**** Building vtk-m"

    # TODO, the gpu version can be different here
    # we only use the cpu version here
    # there are still some issues to run gpu and cpu backend
    # by the same binary? the gpu is dorced to be used anyway?

    cmake -B ${VTKM_BUILD_DIR} -S ${VTKM_SRC_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DVTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON \
    -DVTKm_USE_DOUBLE_PRECISION=ON \
    -DVTKm_USE_64BIT_IDS=OFF \
    -DCMAKE_INSTALL_PREFIX=${VTKM_INSTALL_DIR} \
    -DVTKm_ENABLE_MPI=ON \
    -DVTKm_ENABLE_OPENMP=ON \
    -DVTKm_ENABLE_LOGGING=ON \
    -DVTKm_ENABLE_RENDERING=ON \
    -DVTKm_ENABLE_CUDA=ON \
    -DVTKm_ENABLE_TESTING=OFF \
    -DVTKm_ENABLE_RENDERING=OFF \
    -DVTKm_CUDA_Architecture=volta \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DVTKm_ENABLE_KOKKOS=ON \
    -DKokkos_DIR=${kokkos_install_dir}/lib64/cmake/Kokkos \
    -DKokkos_COMPILE_LAUNCHER=${kokkos_install_dir}/bin/kokkos_launch_compiler \
    -DKokkos_NVCC_WRAPPER=${kokkos_install_dir}/bin/nvcc_wrapper \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc

    cmake --build ${VTKM_BUILD_DIR} -j${build_jobs}

    echo "**** Installing vtk-m"
    cmake --install ${VTKM_BUILD_DIR}
fi

echo "====> Installing vtk-m, ok"

echo "====> Installing EasyLinalg"
EASY_LINALG_SRC_DIR="$SOFTWARE_SRC_DIR/EasyLinalg"
EASY_LINALG_INSTALL_DIR="$HERE/../../ucvworklet/linalg/EasyLinalg/"

rm -rf $EASY_LINALG_SRC_DIR
cd $SOFTWARE_SRC_DIR
git clone $EASY_LINALG_REPO

# move include dir to correct place

# clean old dir if it exist
if [ -d $EASY_LINALG_INSTALL_DIR ]; then
    rm -rf $EASY_LINALG_INSTALL_DIR
fi

mkdir -p $EASY_LINALG_INSTALL_DIR

# move files to new dir
cp EasyLinalg/StaticMemTemplate/include/* $EASY_LINALG_INSTALL_DIR
# clean source files
rm -rf $EASY_LINALG_SRC_DIR

echo "====> Installing EasyLinalg, ok"

echo "====> build UCV"
# the only have build dir without the install dir
# the install dir is same with the build dir
UCV_SRC_DIR=$HERE/../../
# use the install dir as the build dir
UCV_INSTALL_DIR="$SOFTWARE_INSTALL_DIR/UCV"

#if [ -d $UCV_INSTALL_DIR ]; then
#    echo "====> skip, $UCV_INSTALL_DIR already exists," \
#             "please remove it if you want to reinstall it"
#else

    cmake -B ${UCV_INSTALL_DIR} -S ${UCV_SRC_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_CUDA=ON \
    -DKokkos_DIR=${kokkos_install_dir}/lib64/cmake/Kokkos \
    -DVTKm_DIR=${VTKM_INSTALL_DIR}/lib/cmake/vtkm-2.0 \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
    
    cd $HERE

    # build and install
    echo "**** Building UCV"
    cmake --build ${UCV_INSTALL_DIR} -j${build_jobs}
#fi

# not sure why the libvtkmdiympi.so is not included during the build process
echo "try to add library path by executing:"
echo "export LD_LIBRARY_PATH=${VTKM_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH}"
