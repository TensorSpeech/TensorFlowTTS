#include "VoxCommon.hpp"
#include "ext/json.hpp"
using namespace nlohmann;

const std::vector<std::string> Text2MelNames = {"FastSpeech2","Tacotron2"};
const std::vector<std::string> VocoderNames = {"Multi-Band MelGAN"};
const std::vector<std::string> RepoNames = {"TensorflowTTS"};

const std::vector<std::string> LanguageNames = {"English","Spanish"};


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

VoiceInfo VoxUtil::ReadModelJSON(const std::string &InfoFilename)
{
    const size_t MaxNoteSize = 80;

    std::ifstream JFile(InfoFilename);
    json JS;

    JFile >> JS;


    JFile.close();

    auto Arch = JS["architecture"];

    ArchitectureInfo CuArch;
    CuArch.Repo = Arch["repo"].get<int>();
    CuArch.Text2Mel = Arch["text2mel"].get<int>();
    CuArch.Vocoder = Arch["vocoder"].get<int>();

    // Now fill the strings
    CuArch.s_Repo = RepoNames[CuArch.Repo];
    CuArch.s_Text2Mel = Text2MelNames[CuArch.Text2Mel];
    CuArch.s_Vocoder = VocoderNames[CuArch.Vocoder];


    uint32_t Lang = JS["language"].get<uint32_t>();
    VoiceInfo Inf{JS["name"].get<std::string>(),
                 JS["author"].get<std::string>(),
                 JS["version"].get<int>(),
                 JS["description"].get<std::string>(),
                 CuArch,
                 JS["note"].get<std::string>(),
                 JS["sarate"].get<uint32_t>(),
                 Lang,
                LanguageNames[Lang],
                 " " + JS["pad"].get<std::string>()}; // Add a space for separation since we directly append the value to the prompt

    if (Inf.Note.size() > MaxNoteSize)
        Inf.Note = Inf.Note.substr(0,MaxNoteSize);

    return Inf;







}
