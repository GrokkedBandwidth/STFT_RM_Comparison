import uhd
import numpy as np
import matplotlib.pyplot as plt
import librosa

usrp = uhd.usrp.MultiUSRP()
sr = 100000
tuning_freq = 853.75e6
samples = usrp.recv_num_samps(1000000, tuning_freq, sr, [0], 30)
iq_signal = np.asanyarray(samples).flatten()

amin = 1e-10
n_fft = 5096
freqs, times, mags = librosa.reassigned_spectrogram(y=np.real(iq_signal), sr=sr,
                                                    n_fft=n_fft)
mags_db = librosa.amplitude_to_db(mags, ref=np.max)
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,
                         hop_length=n_fft//4, ax=ax[0])
ax[0].set(title="Spectrogram", xlabel=None)
ax[0].label_outer()
ax[0].set_yticks(ax[0].get_yticks())
ax[0].set_yticklabels((ax[0].get_yticks() + tuning_freq) / 1e6)
ax[0].set_ylabel("Frequency [MHz]")
ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
ax[1].set_title("Reassigned spectrogram")
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()

