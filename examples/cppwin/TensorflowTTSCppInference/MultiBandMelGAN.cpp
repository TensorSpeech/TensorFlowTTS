#include "MultiBandMelGAN.h"
#include <stdexcept>
#define IF_EXCEPT(cond,ex) if (cond){throw std::invalid_argument(ex);}



bool MultiBandMelGAN::Initialize(const std::string & VocoderPath)
{
	try {
		MelGAN = new Model(VocoderPath);
	}
	catch (...) {
		MelGAN = nullptr;
		return false;

	}
	return true;


}

TFTensor<float> MultiBandMelGAN::DoInference(const TFTensor<float>& InMel)
{
    IF_EXCEPT(!MelGAN, "Tried to infer MB-MelGAN on uninitialized model!!!!")

	// Convenience reference so that we don't have to constantly derefer pointers.
	Model& Mdl = *MelGAN;

	Tensor input_mels{ Mdl,"serving_default_mels" };
	input_mels.set_data(InMel.Data, InMel.Shape);

	Tensor out_audio{ Mdl,"StatefulPartitionedCall" };

	MelGAN->run(input_mels, out_audio);

	TFTensor<float> RetTensor = VoxUtil::CopyTensor<float>(out_audio);

	return RetTensor;


}

MultiBandMelGAN::MultiBandMelGAN()
{
	MelGAN = nullptr;
}


MultiBandMelGAN::~MultiBandMelGAN()
{
    if (MelGAN)
        delete MelGAN;

}
