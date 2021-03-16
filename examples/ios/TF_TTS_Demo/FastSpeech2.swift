//
//  FastSpeech2.swift
//  HelloTensorFlowTTS
//
//  Created by 안창범 on 2021/03/09.
//

import Foundation
import TensorFlowLite

class FastSpeech2 {
    let interpreter: Interpreter
    
    var speakerId: Int32 = 0
    
    var f0Ratio: Float = 1
    
    var energyRatio: Float = 1
    
    init(url: URL) throws {
        var options = Interpreter.Options()
        options.threadCount = 5
        interpreter = try Interpreter(modelPath: url.path, options: options)
    }
    
    func getMelSpectrogram(inputIds: [Int32], speedRatio: Float) throws -> Tensor {
        try interpreter.resizeInput(at: 0, to: [1, inputIds.count])
        try interpreter.allocateTensors()
        
        let data = inputIds.withUnsafeBufferPointer(Data.init)
        try interpreter.copy(data, toInputAt: 0)
        try interpreter.copy(Data(bytes: &speakerId, count: 4), toInputAt: 1)
        var speedRatio = speedRatio
        try interpreter.copy(Data(bytes: &speedRatio, count: 4), toInputAt: 2)
        try interpreter.copy(Data(bytes: &f0Ratio, count: 4), toInputAt: 3)
        try interpreter.copy(Data(bytes: &energyRatio, count: 4), toInputAt: 4)

        let t0 = Date()
        try interpreter.invoke()
        print("fastspeech2: \(Date().timeIntervalSince(t0))s")
        
        return try interpreter.output(at: 1)
    }
}
