#include "VoxCommon.h"

void VoxUtil::ExportWAV(const std::string & Filename, const std::vector<float>& Data, unsigned SampleRate) {
	AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);


	Buffer[0] = Data;
	size_t BufSz = Data.size();


	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(32);
	File.setSampleRate(SampleRate);

	File.save(Filename, AudioFileFormat::Wave);
}
