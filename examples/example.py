import single_photon_extraction as spe
import single_photon_extraction.plot
import numpy as np
import sebastians_matplotlib_addons as splt


PLOT = True
PLOT_FIGSTYLE = {"rows": 720, "cols": 1920, "fontsize": 1}
PLOT_AXSPAN = [0.1, 0.15, 0.85, 0.8]


prng = np.random.Generator(np.random.MT19937(seed=0))
NUM_SAMPLES = 2 * 1000

NSB_RATE = 10e6

ANALOG_CONFIG = {
    "periode": 0.5e-9,
    "bandwidth": 0.5 * 83.3e6,
}

PULSE_CONFIG = {
    "amplitude": 1.0,
    "amplitude_std": 0.0,
    "decay_time": 50e-9,
    "decay_time_std": 0.0,
}

# make pulse template
# ===================
lowpass_kernel = spe.signal.make_lowpass_kernel(
    periode=ANALOG_CONFIG["periode"],
    lowpass_cutoff_frequency=ANALOG_CONFIG["bandwidth"],
)
pt_offset_num_analog_samples = int(len(lowpass_kernel) // 2)
pt_offset_time = pt_offset_num_analog_samples * ANALOG_CONFIG["periode"]
pt_pulse_start = spe.signal.make_pulse(
    periode=ANALOG_CONFIG["periode"],
    pulse_amplitude=PULSE_CONFIG["amplitude"],
    pulse_decay_time=PULSE_CONFIG["decay_time"],
)
pt_pulse = np.zeros(len(pt_pulse_start) + pt_offset_num_analog_samples)

pt_pulse = spe.signal.add_first_to_second_at(
    f1=pt_pulse_start,
    f2=pt_pulse,
    injection_slices=[pt_offset_num_analog_samples],
)
pt_analog = spe.signal.make_analog_output(
    periode=ANALOG_CONFIG["periode"],
    perfect=pt_pulse,
    lowpass_cutoff_frequency=ANALOG_CONFIG["bandwidth"],
)
ANALOG_PULSE_AMPLITUDE = np.max(pt_analog)

ADC_CONFIG = {
    "skips": 24,
    "amplitude_min": -0.8 * ANALOG_PULSE_AMPLITUDE,
    "amplitude_max": 12.0 * ANALOG_PULSE_AMPLITUDE,
    "amplitude_noise": 0.8 * 0.1,
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

pt_adc = spe.signal.make_adc_output(
    analog=pt_analog,
    skips=ADC_CONFIG["skips"],
    amplitude_noise=0.0,
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=ADC_CONFIG["num_bits"],
    prng=prng,
)

fpga_skips = ADC_CONFIG["skips"] // FPGA_CONFIG["adc_repeats"]
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
    # plot possible phase shift
    # -------------------------
    _time_start = -pt_offset_num_analog_samples * ANALOG_CONFIG["periode"]
    _adc_periode = ANALOG_CONFIG["periode"] * ADC_CONFIG["skips"]

    fig = splt.figure({"rows": 1080, "cols": 1920, "fontsize": 1})
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    num_phases = 3
    phases = np.linspace(0.0, _adc_periode, num_phases, endpoint=False)
    _phase_shift_in_num_analog_samples = ADC_CONFIG["skips"] / num_phases
    assert _phase_shift_in_num_analog_samples % 1 == 0
    _phase_shift_in_num_analog_samples = int(
        _phase_shift_in_num_analog_samples
    )

    for ii in range(len(phases)):
        yoff = 1 * ii
        pps_analog_ts = spe.signal.make_timeseries(
            num_samples=len(pt_analog),
            periode=ANALOG_CONFIG["periode"],
            time_start=phases[ii],
        )
        pps_adc = spe.signal.make_adc_output(
            analog=np.roll(pt_analog, _phase_shift_in_num_analog_samples * ii),
            skips=ADC_CONFIG["skips"],
            amplitude_noise=0.0,
            amplitude_min=ADC_CONFIG["amplitude_min"],
            amplitude_max=ADC_CONFIG["amplitude_max"],
            num_bits=ADC_CONFIG["num_bits"],
            prng=prng,
        )
        pps_adc_lvl = spe.signal.to_analog_level(
            pps_adc,
            amplitude_min=ADC_CONFIG["amplitude_min"],
            amplitude_max=ADC_CONFIG["amplitude_max"],
            num_bits=ADC_CONFIG["num_bits"],
        )
        pps_adc_ts = spe.signal.make_timeseries(
            num_samples=len(pps_adc), periode=_adc_periode, time_start=0.0
        )
        ax.plot(pps_analog_ts + _time_start, pt_analog + yoff, "k", alpha=0.25)
        ax.step(pps_adc_ts + _time_start, pps_adc_lvl + yoff, "k")
        ax.plot([phases[ii], phases[ii]], [yoff, yoff + 1], "k:", alpha=0.25)

    ax.set_xlabel("time / s")
    ax.set_ylabel("amplitude / 1")
    ax.set_xlim(
        [_time_start + _adc_periode * 0, _time_start + _adc_periode * 8]
    )
    fig.savefig("pulse_template_adc_various_phases.jpg")
    splt.close_figure(fig)


if PLOT:
    _time_start = -pt_offset_num_analog_samples * ANALOG_CONFIG["periode"]
    _ts = spe.signal.make_timeseries(
        num_samples=len(pt_pulse),
        periode=ANALOG_CONFIG["periode"],
        time_start=_time_start,
    )
    _ampl_1_over_e = 1.0 / np.exp(1)
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    ax.plot(
        _ts, pt_pulse, "k:",
    )
    ax.plot(
        _ts, pt_analog, "k",
    )
    ax.fill_between(
        x=[0.0, PULSE_CONFIG["decay_time"]],
        y1=[0, 0],
        y2=[_ampl_1_over_e, _ampl_1_over_e],
        hatch="///",
        facecolor="white",
    )
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([min(_ts), max(_ts)])
    ax.set_xlabel("time / s")
    ax.set_ylabel("amplitude / 1")
    fig.savefig("pulse_template_perfect.jpg")
    splt.close_figure(fig)


# plot analog bandwidth
# ---------------------
if PLOT:
    sp_N = 1000 * 1000

    # a delta-function
    sp_ts = spe.signal.make_timeseries(
        num_samples=sp_N, periode=ANALOG_CONFIG["periode"]
    )
    sp_perfect = np.zeros(sp_N)
    sp_perfect[sp_N // 2] = 1

    sp_analog = spe.signal.make_analog_output(
        periode=ANALOG_CONFIG["periode"],
        perfect=sp_perfect,
        lowpass_cutoff_frequency=ANALOG_CONFIG["bandwidth"],
    )

    sp_freq, sp_spec_analog = spe.signal.power_spectrum(
        periode=ANALOG_CONFIG["periode"], sig_vs_t=sp_analog
    )

    sp_f_sample_adc = 1 / ANALOG_CONFIG["periode"] / ADC_CONFIG["skips"]
    sp_f_nyq_adc = 0.5 * sp_f_sample_adc

    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    ax.step(sp_freq, sp_spec_analog, "k")
    ax.axvline(sp_f_sample_adc, color="k", alpha=0.33, linestyle="--")
    ax.axvline(sp_f_nyq_adc, color="k", alpha=0.33, linestyle=":")
    ax.axhline(0.5, color="k", alpha=0.33, linestyle="--")
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
    analog_config=ANALOG_CONFIG,
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
    ax.set_xlabel("time / s")
    ax.set_ylabel("amplitude / 1")
    fig.savefig("event.jpg")
    splt.close_figure(fig)


pt_offset_num_fpga_samples = pt_offset_num_analog_samples / (
    ADC_CONFIG["skips"] / FPGA_CONFIG["adc_repeats"]
)
pt_offset_num_fpga_samples = int(pt_offset_num_fpga_samples)

FPGA_sig_vs_t = spe.signal.to_analog_level(
    digital=event["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=FPGA_CONFIG["num_bits"],
)
ramp_slices = 100
ramp = np.linspace(0, 1, ramp_slices)
FPGA_sig_vs_t[0:ramp_slices] = FPGA_sig_vs_t[0:ramp_slices] * ramp

FPGA_pulse_rising_edge_template = spe.signal.to_analog_level(
    digital=ptemp["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    num_bits=FPGA_CONFIG["num_bits"],
)
FPGA_pulse_rising_edge_template = FPGA_pulse_rising_edge_template[
    0 : 2 * pt_offset_num_fpga_samples
]
if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    ax.step(
        spe.signal.make_timeseries(
            len(FPGA_pulse_rising_edge_template), periode=FPGA_PERIODE,
        ),
        FPGA_pulse_rising_edge_template,
        "k",
    )
    ax.set_xlabel("time / s")
    fig.savefig("FPGA_pulse_rising_edge_template.jpg")
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
            len(FPGA_sub_pulse_template), periode=FPGA_PERIODE
        ),
        FPGA_sub_pulse_template,
        "k",
    )
    ax.set_xlabel("time / s")
    fig.savefig("FPGA_spe_sub_pulse_template.jpg")
    splt.close_figure(fig)

ex_config = {
    "min_amplitude_to_subtract_from": -0.1 * ANALOG_PULSE_AMPLITUDE,
    "pulse_rising_edge_template": FPGA_pulse_rising_edge_template,
    "subtraction_pulse_template": FPGA_sub_pulse_template,
    "num_subtraction_offset_slices": int(
        len(FPGA_pulse_rising_edge_template) * 0.85
    ),
    "stage_thresholds": 0.5 * ANALOG_PULSE_AMPLITUDE * np.ones(10),
    "num_cooldown_slices": len(FPGA_sub_pulse_template),
}

ex_reco_arrival_times, ex_debug = spe.extractors.iterative_subtraction.apply(
    sampling_periode=FPGA_PERIODE, sig=FPGA_sig_vs_t, config=ex_config
)

if PLOT:
    for stage in range(len(ex_config["stage_thresholds"])):
        dbg = ex_debug["stage_debugs"][stage]

        fig = splt.figure(PLOT_FIGSTYLE)
        ax = splt.add_axes(fig, PLOT_AXSPAN)
        ax.step(dbg["sig"], "k", alpha=0.4)
        ax.plot(dbg["sig_conv_pulse"], "k", alpha=0.1)
        ax.step(dbg["extraction_candidates"], "b", alpha=0.1)
        ax.step(dbg["extraction"], "r", alpha=0.1)
        ax.step(
            dbg["cooldown"] / ex_config["num_cooldown_slices"], "g", alpha=1
        )
        ax.set_ylim([-0.2, 1.2])
        ax.set_xlabel("time / fpga-samples")
        ax.set_ylabel("amplitude")
        fig.savefig("spe_{:06d}.jpg".format(stage))
        splt.close_figure(fig)

event["reco_arrival_times"] = ex_reco_arrival_times

performance = []
for time_delta in FPGA_PERIODE * np.arange(25):
    p = spe.benchmark(
        reco_times=event["reco_arrival_times"],
        true_times=event["true_arrival_times"],
        time_delta=time_delta,
    )
    p["time_delta"] = time_delta
    ana = spe.analyse_benchmark(tp=p["tp"], tn=["tn"], fp=p["fp"], fn=p["fn"])
    for key in ana:
        p[key] = ana[key]
    performance.append(p)

if PLOT:
    fig = splt.figure(PLOT_FIGSTYLE)
    ax = splt.add_axes(fig, PLOT_AXSPAN)
    spe.plot.ax_add_event(ax=ax, event=event)
    ax.set_xlabel("time / s")
    ax.set_ylabel("amplitude / 1")
    fig.savefig("final.jpg")
    splt.close_figure(fig)
