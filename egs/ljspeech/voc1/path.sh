# cuda related
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# path related
export PRJ_ROOT="${PWD}/../../.."

export PATH="${PATH}:${PRJ_ROOT}/utils"

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
