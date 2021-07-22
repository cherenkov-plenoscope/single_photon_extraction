import single_photon_extraction as spe
import single_photon_extraction.plot
import numpy as np
import sebastians_matplotlib_addons as splt


PLOT = True
PLOT_FIGSTYLE = {"rows": 720, "cols": 1920, "fontsize": 1}
PLOT_AXSPAN = [0.1, 0.15, 0.85, 0.8]


prng = np.random.Generator(np.random.MT19937(seed=0))
NUM_SAMPLES = 4 * 1000

NSB_RATE = 5e6

ANALOG_CONFIG = {
    "periode": 0.5e-9,
    "bandwidth": 83.3e6,
}

PULSE_CONFIG = {
    "amplitude": 1.0,
    "amplitude_std": 0.0,
    "decay_time": 50e-9,
    "decay_time_std": 0.0,
}

ADC_CONFIG = {
    "skips": 24,
    "amplitude_min": -0.8,
    "amplitude_max": 12.0,
    "amplitude_noise": 0.05,
    "num_bits": 8,
}

FPGA_CONFIG = {
    "adc_repeats": 6,
    "num_bits": 10,
    "kernel": [],
}

FPGA_PERIODE = (
    ANALOG_CONFIG["periode"] * ADC_CONFIG["skips"] / FPGA_CONFIG["adc_repeats"]
)


# make pulse template
# ===================

pt_pulse = spe.signal.make_pulse(
    periode=ANALOG_CONFIG["periode"],
    pulse_amplitude=PULSE_CONFIG["amplitude"],
    pulse_decay_time=PULSE_CONFIG["decay_time"],
)
pt_perfect = spe.signal.make_timeseries(
    num_samples=len(pt_pulse) + 200,
    periode=ANALOG_CONFIG["periode"]
)
pt_perfect = spe.signal.add_first_to_second_at(
    f1=pt_pulse, f2=pt_perfect, injection_slices=[200],
)
pt_analog = spe.signal.make_analog_output(
    periode=ANALOG_CONFIG["periode"],
    perfect=pt_perfect,
    lowpass_cutoff_frequency=ANALOG_CONFIG["bandwidth"],
)
pt_adc = spe.signal.make_adc_output(
    analog=pt_analog,
    skips=ADC_CONFIG["skips"],
    amplitude_noise=0.0,
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=ADC_CONFIG["num_bits"],
    prng=prng,
)

fpga_skips = ADC_CONFIG["skips"]//FPGA_CONFIG["adc_repeats"]
pt_fpga = spe.signal.make_adc_output(
    analog=pt_analog,
    skips=fpga_skips,
    amplitude_noise=0.0,
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=FPGA_CONFIG["num_bits"],
    prng=prng,
)

ptemp = {
    "analog": pt_analog,
    "adc": pt_adc,
    "fpga": pt_fpga,
    "config": {
        "analog": ANALOG_CONFIG,
        "adc": ADC_CONFIG,
        "fpga": FPGA_CONFIG,
    },
}

if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    spe.plot.ax_add_event(ax=ax, event=ptemp)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("gain / 1")
    fig.savefig("pulse_template.jpg")
    splt.close_figure(fig)


# plot analog bandwidth
# ---------------------
if PLOT:
    sp_N = 1000*1000

    # a delta-function
    sp_ts = spe.signal.make_timeseries(
        num_samples=sp_N,
        periode=ANALOG_CONFIG["periode"]
    )
    sp_perfect = np.zeros(sp_N)
    sp_perfect[sp_N//2] = 1

    sp_analog = spe.signal.make_analog_output(
        periode=ANALOG_CONFIG["periode"],
        perfect=sp_perfect,
        lowpass_cutoff_frequency=ANALOG_CONFIG["bandwidth"],
    )

    sp_freq, sp_spec_analog = spe.signal.power_spectrum(
        periode=ANALOG_CONFIG["periode"],
        sig_vs_t=sp_analog
    )

    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    ax.step(sp_freq, sp_spec_analog, "k")
    ax.axvline(
        1/ANALOG_CONFIG["periode"]/ADC_CONFIG["skips"],
        color="k",
        alpha=0.33,
        linestyle="--"
    )
    ax.axhline(
        0.5,
        color="k",
        alpha=0.33,
        linestyle="--"
    )
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("gain / 1")
    ax.set_xlim([1e6, 1e9])
    ax.set_ylim([1e-3, 1.1e0])
    ax.loglog()
    fig.savefig("pulse_power_spectrum.jpg")
    splt.close_figure(fig)


# populate time series with nsb make_pulses
# ====================================

event = spe.make_night_sky_background_event(
    num_samples=NUM_SAMPLES,
    analog_periode=ANALOG_CONFIG["periode"],
    analog_bandwidth=ANALOG_CONFIG["bandwidth"],
    pulse_config=PULSE_CONFIG,
    adc_config=ADC_CONFIG,
    fpga_config=FPGA_CONFIG,
    nsb_rate=NSB_RATE,
    prng=prng,
)

if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    spe.plot.ax_add_event(ax=ax, event=event)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("gain / 1")
    fig.savefig("event.jpg")
    splt.close_figure(fig)


MIN_BASELINE_AMPLITUDE = -0.1 * PULSE_CONFIG["amplitude"]
OFFSET_SLICES = 60
GRAD_THRESHOLDS = 0.4 * np.ones(10)


FPGA_sig_vs_t = spe.signal.to_analog_level(
    digital=event["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=FPGA_CONFIG["num_bits"],
)
ramp_slices = 100
ramp = np.linspace(0, 1, ramp_slices)
FPGA_sig_vs_t[0:ramp_slices] = FPGA_sig_vs_t[0:ramp_slices] * ramp

FPGA_pulse_template = spe.signal.to_analog_level(
    digital=ptemp["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=FPGA_CONFIG["num_bits"],
)[5 * FPGA_CONFIG["adc_repeats"] : 11 * FPGA_CONFIG["adc_repeats"]]
if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    ax.step(
        spe.signal.make_timeseries(
            len(FPGA_pulse_template),
            periode=FPGA_PERIODE,
        ),
        FPGA_pulse_template,
        "k",
    )
    ax.set_xlabel("time / s")
    fig.savefig("FPGA_pulse_template.jpg")
    splt.close_figure(fig)


FPGA_sub_pulse_template = -1.0 * spe.signal.to_analog_level(
    digital=ptemp["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=FPGA_CONFIG["num_bits"],
)
if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    ax.step(
        spe.signal.make_timeseries(
            len(FPGA_sub_pulse_template),
            periode=FPGA_PERIODE
        ),
        FPGA_sub_pulse_template,
        "k",
    )
    ax.set_xlabel("time / s")
    fig.savefig("FPGA_spe_sub_pulse_template.jpg")
    splt.close_figure(fig)

COOLDOWN_SLICES = len(FPGA_sub_pulse_template)



recos = []
remain_sig_vs_t = FPGA_sig_vs_t.copy()

for stage in range(len(GRAD_THRESHOLDS)):
    next_sig_vs_t, reco_arrival_slices, debug = spe.extractors.iterative_subtraction.one_stage(
        sig_vs_t=remain_sig_vs_t,
        min_baseline_amplitude=MIN_BASELINE_AMPLITUDE,
        pulse_template=FPGA_pulse_template,
        sub_pulse_template=FPGA_sub_pulse_template,
        sub_offset_slices=OFFSET_SLICES,
        grad_threshold=GRAD_THRESHOLDS[stage],
        cooldown_slices=COOLDOWN_SLICES,
    )
    recos.append(reco_arrival_slices)

    if PLOT:
        fig = splt.figure(PLOT_FIGSTYLE)
        ax = splt.add_axes(fig, PLOT_AXSPAN)
        ax.plot(remain_sig_vs_t, "k", alpha=0.2)
        ax.plot(debug["sig_conv_pulse"], "k:")
        ax.plot(debug["extraction_candidates"], "b", alpha=0.1)
        ax.plot(debug["extraction"], "r", alpha=0.1)
        ax.set_ylim([-0.2, 1.2])
        ax.set_xlabel("time / samples")
        ax.set_ylabel("amplitude")
        fig.savefig("spe_{:06d}.jpg".format(stage))
        splt.close_figure(fig)

    remain_sig_vs_t = next_sig_vs_t

reco_arrival_fpga_slices = np.concatenate(recos)
reco_arrival_fpga_slices += int(0.8 * OFFSET_SLICES)


event["reco_arrival_times"] = reco_arrival_fpga_slices * FPGA_PERIODE
event["reco_arrival_times"] = np.sort(event["reco_arrival_times"])


performance = []
for time_delta in FPGA_PERIODE*np.arange(25):
    p = spe.benchmark(
        reco_times=event["reco_arrival_times"],
        true_times=event["true_arrival_times"],
        time_delta=time_delta,
    )
    p["time_delta"] = time_delta
    performance.append(p)

if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    spe.plot.ax_add_event(ax=ax, event=event)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("gain / 1")
    fig.savefig("final.jpg")
    splt.close_figure(fig)
