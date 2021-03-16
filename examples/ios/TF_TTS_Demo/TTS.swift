//
//  TTS.swift
//  TF TTS Demo
//
//  Created by 안창범 on 2021/03/16.
//

import Foundation
import AVFoundation

public class TTS {
    var rate: Float = 1.0
    
    private let fastSpeech2 = try! FastSpeech2(url: Bundle.main.url(forResource: "fastspeech2_quan", withExtension: "tflite")!)
    
    private let mbMelGan = try! MBMelGan(url: Bundle.main.url(forResource: "mb_melgan", withExtension: "tflite")!)

    /// Mel spectrogram hop size
    public let hopSize = 256
    
    /// Vocoder sample rate
    let sampleRate = 22_050

    private let sampleBufferRenderSynchronizer = AVSampleBufferRenderSynchronizer()

    private let sampleBufferAudioRenderer = AVSampleBufferAudioRenderer()

    init() {
        sampleBufferRenderSynchronizer.addRenderer(sampleBufferAudioRenderer)
    }

    public func speak(string: String) {
        let input_ids = text_to_sequence(string)
        
        do {
            let melSpectrogram = try fastSpeech2.getMelSpectrogram(inputIds: input_ids, speedRatio: 2 - rate)
            
            let data = try mbMelGan.getAudio(input: melSpectrogram)
            print(data)
            
            let blockBuffer = try CMBlockBuffer(length: data.count)
            try data.withUnsafeBytes { try blockBuffer.replaceDataBytes(with: $0) }
            
            let audioStreamBasicDescription = AudioStreamBasicDescription(mSampleRate: Float64(sampleRate), mFormatID: kAudioFormatLinearPCM, mFormatFlags: kAudioFormatFlagIsFloat, mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
            
            let formatDescription = try CMFormatDescription(audioStreamBasicDescription: audioStreamBasicDescription)
            
            let delay: TimeInterval = 1
            
            let sampleBuffer = try CMSampleBuffer(dataBuffer: blockBuffer,
                                                  formatDescription: formatDescription,
                                                  numSamples: data.count / 4,
                                                  presentationTimeStamp: sampleBufferRenderSynchronizer.currentTime()
                                                    + CMTime(seconds: delay, preferredTimescale: CMTimeScale(sampleRate)),
                                                  packetDescriptions: [])
            
            sampleBufferAudioRenderer.enqueue(sampleBuffer)
            
            sampleBufferRenderSynchronizer.rate = 1
        }
        catch {
            print(error)
        }
    }

    lazy var eos_id = symbolIds["eos"]!
    
    lazy var symbolIds: [String: Int32] = try! loadMapper(url: Bundle.main.url(forResource: "ljspeech_mapper", withExtension: "json")!).symbol_to_id
    
    public func text_to_sequence(_ text: String) -> [Int32] {
        var sequence: [Int32] = []
        sequence += symbols_to_sequence(text)
        sequence.append(eos_id)
        return sequence
    }
    
    func symbols_to_sequence(_ text: String) -> [Int32] {
        return text.unicodeScalars.compactMap { symbolIds[String($0)] }
    }

    func loadMapper(url: URL) throws -> Mapper {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Mapper.self, from: data)
    }
}

extension TTS: ObservableObject {
    
}

public struct Mapper: Codable {
    public let symbol_to_id: [String: Int32]
    public let id_to_symbol: [String: String]
    public let speakers_map: [String: Int32]
    public let processor_name: String
}
