import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from marmots import antenna, galactic_noise



MOUNTAIN_TEMPERATURE = 300
TRACE_LENGTH = 2048 * 5
location = (-35.10, -69.55)  # Auger in lat, long (degrees)

CONFIGS = [
    {
        "name": "30 - 80 (Beacon)",
        "interpolation_frequencies": np.linspace(30, 80, 20),
        "sampling_rate": 200,
        "passband": [30, 80],
        "n_side": 8,
        "ele_temps": [300, 100],
        "ants": [
            antenna.Detector(model="prototype", freqs=np.linspace(30, 80, 10)),
            antenna.Detector(model="prototype", freqs=np.linspace(30, 80, 10), rot=90),
        ],
        "labels": ["Beacon", "Beacon + 90deg", "Beacon isotrop"],
    },
]

hours = np.arange(0, 24, 2)

fig, axes = plt.subplots(1, len(CONFIGS), figsize=(12, 8), sharex=True, sharey=True)


axes = [axes]

for cfg, ax in zip(CONFIGS, axes):
    ants = cfg["ants"]
    labels = cfg["labels"]
    passband = cfg["passband"]
    channel_ids = list(range(len(ants)))

    noise_sim = galactic_noise.GalacticNoiseSimulator()
    noise_sim.begin(
        n_side=cfg["n_side"],
        freq_range=passband,
        caching=True,
    )

    data = []
    for h in hours:

        time = dt.datetime(2024, 1, 1, h, 0, 0)

        waveforms = noise_sim.run(
            TRACE_LENGTH, cfg['sampling_rate'],
            antennas=ants, time=time,
            location=location, passband=passband,
            mountain_elevation=2,
            mountain_temperature=MOUNTAIN_TEMPERATURE,
        )

        data.append(waveforms)

    stds = np.std(np.array(data), axis=-1) * 1e6  # convert V -> muV

    for idx, (ele, label) in enumerate(zip(stds.T, labels)):
        ax.plot(hours, ele, label=label, color=f"C{idx}")

    for t_ele, ls in zip(cfg["ele_temps"], ["--", ":"]):
        vrms = galactic_noise.calculate_vrms_from_temperature(t_ele, bandwidth=passband[1] - passband[0])  * 1e6  # V -> mV

        ax.axhline(vrms, color="k", linestyle=ls, lw=1, label=f"RMS {t_ele} K")
        for idx, ele in enumerate(stds.T):
            ax.plot(hours, np.sqrt(ele**2 + vrms**2), ls=ls, color=f"C{idx}")


    ax.legend(fontsize=7, title=cfg["name"])

# axes.flatten()[-1].set_visible(False)

fig.supxlabel("Time [h]")
fig.supylabel(r"RMS [$\mu$V]", x=0)

fig.suptitle(f"Sky + {MOUNTAIN_TEMPERATURE:.0f} K mountains (2° elevation)")
fig.tight_layout()
fig.savefig(f"galactic_noise_all_{MOUNTAIN_TEMPERATURE:.0f}K.png")
plt.show()
