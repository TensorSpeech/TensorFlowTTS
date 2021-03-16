//
//  MBMelGAN.swift
//  HelloTensorFlowTTS
//
//  Created by 안창범 on 2021/03/09.
//

import Foundation
import TensorFlowLite

class MBMelGan {
    let interpreter: Interpreter
    
    init(url: URL) throws {
        var options = Interpreter.Options()
        options.threadCount = 5
        interpreter = try Interpreter(modelPath: url.path, options: options)
    }
    
    func getAudio(input: Tensor) throws -> Data {
        try interpreter.resizeInput(at: 0, to: input.shape)
        try interpreter.allocateTensors()
        
        try interpreter.copy(input.data, toInputAt: 0)

        let t0 = Date()
        try interpreter.invoke()
        print("mbmelgan: \(Date().timeIntervalSince(t0))s")

        return try interpreter.output(at: 0).data
    }
}
