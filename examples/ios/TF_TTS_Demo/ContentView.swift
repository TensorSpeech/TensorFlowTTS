//
//  ContentView.swift
//  TF TTS Demo
//
//  Created by 안창범 on 2021/03/16.
//

import SwiftUI

struct ContentView: View {
    @StateObject var tts = TTS()
    
    @State var text = "The Rhodes Must Fall campaigners said the announcement was hopeful, but warned they would remain cautious until the college had actually carried out the removal."
    
    var body: some View {
        VStack {
            TextEditor(text: $text)
            Button {
                tts.speak(string: text)
            } label: {
                Label("Speak", systemImage: "speaker.1")
            }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
