#pragma once
/*
 VoxCommon.hpp : Defines common data structures and constants to be used with TensorVox 
*/
#include <iostream>
#include <vector>
#include "ext/AudioFile.hpp"
#include "ext/CppFlow/include/Tensor.h"
#include <stdexcept>

#define IF_RETURN(cond,ret) if (cond){return ret;}
#define VX_IF_EXCEPT(cond,ex) if (cond){throw std::invalid_argument(ex);}


template<typename T>
struct TFTensor {
	std::vector<T> Data;
	std::vector<int64_t> Shape;
	size_t TotalSize;

};



namespace VoxUtil {
	template<typename F>
	TFTensor<F> CopyTensor(Tensor& InTens) 
	{
		std::vector<F> Data = InTens.get_data<F>();
		std::vector<int64_t> Shape = InTens.get_shape();
		size_t TotalSize = 1;
		for (const int64_t& Dim : Shape)
			TotalSize *= Dim;

		return TFTensor<F>{Data, Shape, TotalSize};


	}

	template<typename V>
	bool FindInVec(V In, const std::vector<V>& Vec, size_t& OutIdx, size_t start = 0) {
		for (size_t xx = start;xx < Vec.size();xx++)
		{
			if (Vec[xx] == In) {
				OutIdx = xx;
				return true;

			}

		}


		return false;

	}

	void ExportWAV(const std::string& Filename, const std::vector<float>& Data, unsigned SampleRate);
}
