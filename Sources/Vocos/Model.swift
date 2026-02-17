import Foundation
import Hub
import MLX
import MLXFFT
import MLXNN
import MLXRandom

// mel spec

func hanning(_ size: Int) -> MLXArray {
    let window = (0 ..< size).map { 0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(size - 1))) }
    return MLXArray(converting: window)
}

class MelSpectrogramFeatures: Module {
    let sampleRate: Int
    let nFFT: Int
    let hopLength: Int
    let nMels: Int
    let filterbank: MLXArray
    
    init(
        sampleRate: Int = 24000,
        nFFT: Int = 1024,
        hopLength: Int = 256,
        nMels: Int = 100,
        filterbank: MLXArray
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.filterbank = filterbank
    }
    
    func callAsFunction(x: MLXArray) -> MLXArray {
        logMelSpectrogram(audio: x, nMels: nMels, nFFT: nFFT, hopLength: hopLength, filterbank: filterbank)
    }

    func stft(x: MLXArray, window: MLXArray, nperseg: Int, noverlap: Int? = nil, nfft: Int? = nil) -> MLXArray {
        let nfft = nfft ?? nperseg
        let noverlap = noverlap ?? nfft
        let padding = nperseg / 2
        let x = MLX.padded(x, width: IntOrPair(padding))
        let strides = [noverlap, 1]
        let t = (x.shape[0] - nperseg + noverlap) / noverlap
        let shape = [t, nfft]
        let stridedX = MLX.asStrided(x, shape, strides: strides)
        return MLXFFT.rfft(stridedX * window)
    }

    func logMelSpectrogram(audio: MLXArray, nMels: Int = 100, nFFT: Int = 1024, hopLength: Int = 256, filterbank: MLXArray) -> MLXArray {
        let freqs = stft(x: audio, window: hanning(nFFT), nperseg: nFFT, noverlap: hopLength)
        let magnitudes = freqs[0 ..< freqs.shape[0] - 1].abs()
        let melSpec = MLX.matmul(magnitudes, filterbank.T)
        let logSpec = MLX.maximum(melSpec, 1e-5).log()
        return MLX.expandedDimensions(logSpec, axis: 0)
    }
}

// ISTFT head

class ISTFTHead: Module {
    let nFFT: Int
    let hopLength: Int
    
    let out: Linear
    
    init(dim: Int, nFFT: Int, hopLength: Int) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.out = Linear(dim, nFFT + 2)
    }
    
    func callAsFunction(input: MLXArray) -> MLXArray {
        let input = out(input).swappedAxes(1, 2)
        let split = input.split(parts: 2, axis: 1)
        let p = split[1]
        let mag = MLX.exp(split[0])
        
        let x = MLX.cos(p)
        let y = MLX.sin(p).asImaginary()
        let S = mag * (x + y)
        
        let audio = istft(
            x: S.squeezed(axis: 0).swappedAxes(0, 1),
            window: hanning(nFFT),
            nperseg: nFFT,
            noverlap: hopLength,
            nfft: nFFT
        )
        return audio
    }

    func istft(x: MLXArray, window: MLXArray, nperseg: Int = 256, noverlap: Int? = nil, nfft: Int? = nil) -> MLXArray {
        let nfft = nfft ?? nperseg
        let noverlap = noverlap ?? nfft
        let t = [(x.shape[0] - 1) * noverlap + nperseg]
        let reconstructed = MLX.zeros(t)
        let window_sum = MLX.zeros(t)
        
        for i in 0 ..< x.shape[0] {
            // inverse FFT of each frame
            let frame_time = MLXFFT.irfft(x[i])
            
            // get the position in the time-domain signal to add the frame
            let start = i * noverlap
            let end = start + nperseg
            
            // overlap-add the inverse transformed frame, scaled by the window
            reconstructed[start ..< end] = reconstructed[start ..< end] + (frame_time * window)
            window_sum[start ..< end] = window_sum[start ..< end] + window
        }
        
        // normalize by the sum of the window values
        return MLX.where(window_sum .!= 0, reconstructed / window_sum, reconstructed)
    }
}

// ConvNeXT blocks

open class GroupableConv1d: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: Int
    public let groups: Int
    public let stride: Int
    
    convenience init(_ inputChannels: Int, _ outputChannels: Int, kernelSize: Int, padding: Int, groups: Int) {
        self.init(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: kernelSize, padding: padding, groups: groups)
    }
    
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))
        
        self.weight = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups]
        )
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.stride = stride
        self.groups = groups
    }
    
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

class ConvNeXtBlock: Module {
    let norm: LayerNorm
    let dwconv: GroupableConv1d
    let pwconv1: Linear
    let act: GELU
    let pwconv2: Linear
    let gamma: MLXArray
    
    init(
        dim: Int,
        intermediateDim: Int,
        layerScaleInitValue: Float
    ) {
        self.dwconv = GroupableConv1d(dim, dim, kernelSize: 7, padding: 3, groups: dim)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        self.pwconv1 = Linear(dim, intermediateDim)
        self.act = GELU()
        self.pwconv2 = Linear(intermediateDim, dim)
        self.gamma = layerScaleInitValue * MLXArray.ones([dim])
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = dwconv(x)
        x = norm(x)
        x = pwconv1(x)
        x = act(x)
        x = pwconv2(x)
        x = gamma * x
        x = residual + x
        return x
    }
}

// backbone

class VocosBackbone: Module {
    var embed: Conv1d
    var norm: LayerNorm
    var convnext: [ConvNeXtBlock]
    let final_layer_norm: LayerNorm
    
    init(
        inputChannels: Int,
        dim: Int,
        intermediateDim: Int,
        numLayers: Int,
        layerScaleInitValue: Float? = nil
    ) {
        self.embed = Conv1d(inputChannels: inputChannels, outputChannels: dim, kernelSize: 7, padding: 3)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        let layerScaleInitValue = layerScaleInitValue ?? 1 / Float(numLayers)
        self.convnext = (0 ..< numLayers).map { _ in ConvNeXtBlock(dim: dim, intermediateDim: intermediateDim, layerScaleInitValue: layerScaleInitValue) }
        self.final_layer_norm = LayerNorm(dimensions: dim, eps: 1e-6)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = embed(x)
        x = norm(x)
        for convBlock in convnext {
            x = convBlock(x)
        }
        x = final_layer_norm(x)
        return x
    }
}

// main class

public class Vocos: Module {
    enum VocosError: Error {
        case unableToLoadModel
    }
    
    let feature_extractor: MelSpectrogramFeatures
    let backbone: VocosBackbone
    let head: ISTFTHead
    
    init(feature_extractor: MelSpectrogramFeatures, backbone: VocosBackbone, head: ISTFTHead) {
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
    }
    
    public func decode(_ featuresInput: MLXArray) -> MLXArray {
        let x = backbone(featuresInput)
        return head(input: x)
    }
    
    public func callAsFunction(_ audioInput: MLXArray) -> MLXArray {
        let features = feature_extractor(x: audioInput)
        return decode(features)
    }
}

// pre-trained models

public extension Vocos {
    static func fromPretrained(repoId: String) async throws -> Vocos {
        let modelDirectoryURL = try await Hub.snapshot(from: repoId, matching: ["*.safetensors", "*.json"])
        return try fromPretrained(modelDirectoryURL: modelDirectoryURL)
    }
    
    static func fromPretrained(modelDirectoryURL: URL) throws -> Vocos {
        let modelURL = modelDirectoryURL.appendingPathComponent("model.safetensors")
        var modelWeights = try loadArrays(url: modelURL)
        
        guard let filterbank = modelWeights.removeValue(forKey: "feature_extractor.filterbank") else {
            throw VocosError.unableToLoadModel
        }
        
        let configURL = modelDirectoryURL.appendingPathComponent("config.json")
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]
        guard let config else {
            throw VocosError.unableToLoadModel
        }
        
        let vocos = try fromConfig(config: config, filterbank: filterbank)
        
        var weights = [String: MLXArray]()
        for (key, value) in modelWeights {
            weights[key] = value
        }
        let parameters = ModuleParameters.unflattened(weights)
        // NOTE: Using .noUnusedKeys because filterbank was extracted from weights
        // and passed to constructor separately (line 269)
        try vocos.update(parameters: parameters, verify: [.noUnusedKeys])
        
        return vocos
    }
    
    static func fromConfig(config: [String: Any], filterbank: MLXArray) throws -> Vocos {
        var featureExtractor: MelSpectrogramFeatures?
        
        if let featureExtractorConfig = config["feature_extractor"] as? [String: Any],
           let initArgs = featureExtractorConfig["init_args"] as? [String: Any],
           let sampleRate = initArgs["sample_rate"] as? Int,
           let nFFT = initArgs["n_fft"] as? Int,
           let hopLength = initArgs["hop_length"] as? Int,
           let nMels = initArgs["n_mels"] as? Int {
            featureExtractor = MelSpectrogramFeatures(
                sampleRate: sampleRate,
                nFFT: nFFT,
                hopLength: hopLength,
                nMels: nMels,
                filterbank: filterbank
            )
        }
        
        var backbone: VocosBackbone?
        
        if let backboneConfig = config["backbone"] as? [String: Any],
           let initArgs = backboneConfig["init_args"] as? [String: Any],
           let inputChannels = initArgs["input_channels"] as? Int,
           let dim = initArgs["dim"] as? Int,
           let intermediateDim = initArgs["intermediate_dim"] as? Int,
           let numLayers = initArgs["num_layers"] as? Int {
            backbone = VocosBackbone(
                inputChannels: inputChannels,
                dim: dim,
                intermediateDim: intermediateDim,
                numLayers: numLayers
            )
        }
        
        var head: ISTFTHead?
        
        if let headConfig = config["head"] as? [String: Any],
           let initArgs = headConfig["init_args"] as? [String: Any],
           let dim = initArgs["dim"] as? Int,
           let nFFT = initArgs["n_fft"] as? Int,
           let hopLength = initArgs["hop_length"] as? Int {
            head = ISTFTHead(
                dim: dim,
                nFFT: nFFT,
                hopLength: hopLength
            )
        }
        
        guard let featureExtractor, let backbone, let head else {
            throw VocosError.unableToLoadModel
        }
        
        return Vocos(feature_extractor: featureExtractor, backbone: backbone, head: head)
    }
}
