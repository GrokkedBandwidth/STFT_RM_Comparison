import uhd
import numpy as np
import matplotlib.pyplot as plt
import librosa

usrp = uhd.usrp.MultiUSRP()
sr = 0.2e6
tuning_freq = 145.95e6
samples = usrp.recv_num_samps(1000000, tuning_freq, sr, [0], 0)
iq_signal = np.asanyarray(samples).flatten()

n_fft = 5096
freqs, times, mags = librosa.reassigned_spectrogram(y=np.real(iq_signal),
                                                    sr=sr,
                                                    n_fft=n_fft)
mags_db = librosa.amplitude_to_db(mags, ref=np.max)
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10,8))
img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,
                         hop_length=n_fft//4, ax=ax[0])
ax[0].set(title="Spectrogram", xlabel=None)
ax[0].label_outer()
ax[0].set_yticks(ax[0].get_yticks())
ax[0].set_yticklabels((ax[0].get_yticks() + tuning_freq) / 1e6)
ax[0].set_ylabel("Frequency [MHz]")
ax[1].set_xlabel("Time [ms]")
ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
ax[1].set_title("Reassigned Spectrogram")
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()

