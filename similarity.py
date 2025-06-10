# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# import librosa
# import librosa.display
# from matplotlib import gridspec

# def plot_voice_comparison(original_path, cloned_path, sample_rate=16000):
#     # Load audio files
#     sr1, original_audio = wavfile.read(original_path)
#     sr2, cloned_audio = wavfile.read(cloned_path)

#     # If stereo, take first channel
#     if len(original_audio.shape) > 1:
#         original_audio = original_audio[:, 0]
#     if len(cloned_audio.shape) > 1:
#         cloned_audio = cloned_audio[:, 0]

#     # Check sample rates
#     print(f"Original sample rate: {sr1}, Cloned sample rate: {sr2}")

#     # Trim to shortest length
#     min_len = min(len(original_audio), len(cloned_audio))
#     original_audio = original_audio[:min_len]
#     cloned_audio = cloned_audio[:min_len]

#     # Normalize integer audio to float between -1 and 1
#     def normalize_audio(audio):
#         if audio.dtype.kind in 'iu':  # integer type
#             return audio.astype(np.float32) / np.iinfo(audio.dtype).max
#         else:
#             return audio.astype(np.float32)

#     original_audio = normalize_audio(original_audio)
#     cloned_audio = normalize_audio(cloned_audio)

#     # Calculate pitch (F0) using librosa.pyin
#     f0_orig, _, _ = librosa.pyin(original_audio, 
#                                  fmin=librosa.note_to_hz('C2'), 
#                                  fmax=librosa.note_to_hz('C7'), sr=sample_rate)
#     f0_clone, _, _ = librosa.pyin(cloned_audio, 
#                                  fmin=librosa.note_to_hz('C2'), 
#                                  fmax=librosa.note_to_hz('C7'), sr=sample_rate)

#     # Time axis for audio and pitch
#     time_audio = np.linspace(0, min_len / sample_rate, num=min_len)
#     time_f0 = np.linspace(0, min_len / sample_rate, num=len(f0_orig))

#     # Plot setup
#     fig = plt.figure(figsize=(15, 10), dpi=300)
#     gs = gridspec.GridSpec(3, 2, figure=fig)

#     # Waveform plot - Original and Cloned
#     ax1 = fig.add_subplot(gs[0, :])
#     ax1.plot(time_audio, original_audio, label='Original Voice', alpha=0.7)
#     ax1.plot(time_audio, cloned_audio, label='Cloned Voice', alpha=0.7)
#     ax1.set_title('Waveform Comparison')
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Amplitude')
#     ax1.legend()
#     ax1.grid(True, linestyle='--', alpha=0.5)

#     # Spectrogram - Original
#     ax2 = fig.add_subplot(gs[1, 0])
#     S_orig = librosa.stft(original_audio)
#     S_db_orig = librosa.amplitude_to_db(np.abs(S_orig), ref=np.max)
#     img1 = librosa.display.specshow(S_db_orig, sr=sample_rate, x_axis='time', y_axis='log', ax=ax2, cmap='viridis')
#     ax2.set_title('Original Voice Spectrogram')
#     fig.colorbar(img1, ax=ax2, format="%+2.0f dB")

#     # Spectrogram - Cloned
#     ax3 = fig.add_subplot(gs[1, 1])
#     S_clone = librosa.stft(cloned_audio)
#     S_db_clone = librosa.amplitude_to_db(np.abs(S_clone), ref=np.max)
#     img2 = librosa.display.specshow(S_db_clone, sr=sample_rate, x_axis='time', y_axis='log', ax=ax3, cmap='viridis')
#     ax3.set_title('Cloned Voice Spectrogram')
#     fig.colorbar(img2, ax=ax3, format="%+2.0f dB")

#     # Pitch (F0) - Original and Cloned
#     ax4 = fig.add_subplot(gs[2, :])
#     ax4.plot(time_f0, f0_orig, label='Original Pitch (F0)', alpha=0.8)
#     ax4.plot(time_f0, f0_clone, label='Cloned Pitch (F0)', alpha=0.8)
#     ax4.set_title('Pitch (Fundamental Frequency) Comparison')
#     ax4.set_xlabel('Time (s)')
#     ax4.set_ylabel('Frequency (Hz)')
#     ax4.legend()
#     ax4.grid(True, linestyle='--', alpha=0.5)

#     plt.tight_layout()
#     output_file = 'voice_comparison_full.png'
#     plt.savefig(output_file, bbox_inches='tight')
#     plt.show()
#     print(f"✔ Voice comparison figure saved as {output_file}")

# # Usage example (replace filenames with your files)
# plot_voice_comparison('speaker_orignal.wav', 'anza_clone.wav')





import numpy as np
import matplotlib.pyplot as plt
from resemblyzer import VoiceEncoder, preprocess_wav
from matplotlib.gridspec import GridSpec

def plot_voice_comparison(original_path, cloned_path, sample_rate=16000):
    try:
        # Load and process audio files
        original_wav = preprocess_wav(original_path)
        cloned_wav = preprocess_wav(cloned_path)
        
        # Initialize encoder
        encoder = VoiceEncoder()
        print("Model loaded successfully")

        # Get embeddings and similarity
        original_embed = encoder.embed_utterance(original_wav)
        cloned_embed = encoder.embed_utterance(cloned_wav)
        similarity = np.inner(original_embed, cloned_embed)

        # Create figure with subplots
        fig = plt.figure(figsize=(12, 8), dpi=300)
        gs = GridSpec(2, 2, figure=fig)

        # Plot 1: Waveform Comparison (aligned to shorter length)
        min_length = min(len(original_wav), len(cloned_wav))
        time = np.linspace(0, min_length/sample_rate, num=min_length)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, original_wav[:min_length], label="Original Voice", alpha=0.7, color='#1f77b4')
        ax1.plot(time, cloned_wav[:min_length], label="Cloned Voice", alpha=0.7, color='#ff7f0e')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Waveform Comparison (First {min_length/sample_rate:.2f}s)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Similarity Score
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(['Similarity'], [similarity], color=['#2ca02c'])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Voice Similarity Score')
        ax2.text(0, similarity + 0.02, f"{similarity:.4f}", ha='center', fontweight='bold')

        # Plot 3: Spectrogram (using original voice)
        ax3 = fig.add_subplot(gs[1, 1])
        Pxx, freqs, bins, im = ax3.specgram(original_wav[:min_length], Fs=sample_rate, cmap='viridis')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title('Original Voice Spectrogram')
        fig.colorbar(im, ax=ax3, label='Intensity (dB)')

        plt.tight_layout()
        output_path = 'voice_comparison.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✔ Visualization saved to {output_path}")
        print(f"🔊 Similarity score: {similarity:.4f}")
        
        return similarity, output_path

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None

# Usage
similarity, img_path = plot_voice_comparison(
    original_path="voice_sample.wav",
    cloned_path="anza_clone.wav"
)






