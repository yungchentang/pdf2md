import AppKit
import Foundation
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count >= 2 else {
    fail("usage: vision_ocr.swift IMAGE_PATH [--json]")
}

let imagePath = CommandLine.arguments[1]
let outputJson = CommandLine.arguments.dropFirst(2).contains("--json")
let imageURL = URL(fileURLWithPath: imagePath)

guard let nsImage = NSImage(contentsOf: imageURL) else {
    fail("failed to load image: \(imagePath)")
}

var rect = NSRect(origin: .zero, size: nsImage.size)
guard let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
    fail("failed to create CGImage: \(imagePath)")
}

var recognized: [(text: String, box: CGRect, confidence: Float)] = []
var requestError: Error?

let request = VNRecognizeTextRequest { request, error in
    if let error = error {
        requestError = error
        return
    }
    guard let observations = request.results as? [VNRecognizedTextObservation] else {
        return
    }
    for observation in observations {
        guard let candidate = observation.topCandidates(1).first else {
            continue
        }
        let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            recognized.append((text: text, box: observation.boundingBox, confidence: candidate.confidence))
        }
    }
}

request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fail("vision request failed: \(error)")
}

if let requestError = requestError {
    fail("vision request failed: \(requestError)")
}

if outputJson {
    let payload = recognized.map { item in
        [
            "text": item.text,
            "min_x": item.box.minX,
            "min_y": item.box.minY,
            "max_x": item.box.maxX,
            "max_y": item.box.maxY,
            "confidence": item.confidence,
        ] as [String: Any]
    }
    do {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    } catch {
        fail("failed to encode OCR JSON: \(error)")
    }
    exit(0)
}

let sorted = recognized.sorted { lhs, rhs in
    let yDelta = abs(lhs.box.midY - rhs.box.midY)
    if yDelta > 0.012 {
        return lhs.box.midY > rhs.box.midY
    }
    return lhs.box.minX < rhs.box.minX
}

var lines: [String] = []
var currentLine: [String] = []
var currentY: CGFloat?

for item in sorted {
    if let y = currentY, abs(y - item.box.midY) > 0.012 {
        lines.append(currentLine.joined(separator: " "))
        currentLine = []
        currentY = item.box.midY
    } else if currentY == nil {
        currentY = item.box.midY
    }
    currentLine.append(item.text)
}

if !currentLine.isEmpty {
    lines.append(currentLine.joined(separator: " "))
}

print(lines.joined(separator: "\n"))
