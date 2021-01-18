#pragma once
/*
 VoxCommon.hpp : Defines common data structures and constants to be used with TensorVox 
*/
#include <iostream>
#include <vector>
#include "AudioFile.h"
// #include "ext/CppFlow/include/Tensor.h"
// #include <stdexcept>

#define IF_RETURN(cond,ret) if (cond){return ret;}
#define VX_IF_EXCEPT(cond,ex) if (cond){throw std::invalid_argument(ex);}


template<typename T>
struct TFTensor {
	std::vector<T> Data;
	std::vector<int64_t> Shape;
	size_t TotalSize;
};

namespace VoxUtil {

	void ExportWAV(const std::string& Filename, const std::vector<float>& Data, unsigned SampleRate);
}
