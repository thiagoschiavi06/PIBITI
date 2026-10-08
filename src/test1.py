import pywt
import soundfile as sf

audio, sr = sf.read("src/audio01.wav")

wp = pywt.WaveletPacket(
    data = audio,
    wavelet = "db4",
    mode = "symmetric",
    maxlevel=4
)

nodes = wp.get_level(4, order = "freq")

print("a: ", len(nodes))

for i, node in enumerate(nodes):
    print(i, node.path, len(node.data))