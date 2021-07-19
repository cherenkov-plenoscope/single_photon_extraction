import single_photon_extraction as spe
import single_photon_extraction.plot
import numpy as np
import sebastians_matplotlib_addons as splt


prng = np.random.Generator(np.random.MT19937(seed=0))
PLOT = True
NUM_SAMPLES = 3 * 1000

NSB_RATE = 5e6

ANALOG_CONFIG = {
    "periode": 0.5e-9,
    "bandwidth": 83.3e6,
}

PULSE_CONFIG = {
    "amplitude": 1.0,
    "amplitude_std": 0.0,
    "decay_time": 50e-9,
}

ADC_CONFIG = {
    "skips": 24,
    "amplitude_min": -0.8,
    "amplitude_max": 12.0,
    "noise": 0.05,
}

FPGA_CONFIG = {
    "adc_repeats": 6,
    "kernel": [],
}

ADC_FREQUENCY = (1 / ANALOG_CONFIG["periode"]) / ADC_CONFIG["skips"]

FPGA_FREQUENCY = (
    (1 / ANALOG_CONFIG["periode"])
    / ADC_CONFIG["skips"]
    * FPGA_CONFIG["adc_repeats"]
)


# make pulse template
# ===================

pt_pulse = spe.make_pulse(
    periode=ANALOG_CONFIG["periode"],
    pulse_amplitude=PULSE_CONFIG["amplitude"],
    pulse_decay_time=PULSE_CONFIG["decay_time"],
)
pt_bandwitdh_kernel = spe.make_bell(
    periode=ANALOG_CONFIG["periode"], bell_std=(1 / ANALOG_CONFIG["bandwidth"])
)
pt_analog = spe.make_timeseries(
    num_samples=len(pt_pulse) + 200,
    periode=ANALOG_CONFIG["periode"]
)
pt_analog = spe.add_first_to_second_at(
    f1=pt_pulse, f2=pt_analog, injection_slices=[200],
)
pt_analog = np.convolve(pt_analog, pt_bandwitdh_kernel, mode="same")

pt_adc = spe.make_adc_output(
    analog=pt_analog,
    skips=ADC_CONFIG["skips"],
    noise_amplitude=0.0,
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
    prng=prng,
)

pt_fpga = spe.make_adc_output(
    analog=pt_analog,
    skips=ADC_CONFIG["skips"]//FPGA_CONFIG["adc_repeats"],
    noise_amplitude=0.0,
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
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


spe.plot.plot_event(event=ptemp, path="pulse_template.jpg")

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

spe.plot.plot_event(event, path="event.jpg")

# run extraction ADC-speed
# ========================

offset_slices = 9

sig_vs_t = spe.to_analog_level(
    digital=event["adc"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
)
sig_vs_t[0:offset_slices] = 0.0

pulse_template = spe.to_analog_level(
    digital=ptemp["adc"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
)[3:11]
spe.plot.plot_extraction_state(
    dig=pulse_template,
    ADC_FREQUENCY=ADC_FREQUENCY,
    truth=event["true_arrival_times"],
    path="pulse_template.jpg",
)

sub_pulse_template = -1.0 * spe.to_analog_level(
    digital=ptemp["adc"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
)
spe.plot.plot_extraction_state(
    dig=sub_pulse_template,
    ADC_FREQUENCY=ADC_FREQUENCY,
    truth=event["true_arrival_times"],
    path="spe_sub_pulse_template.jpg",
)


def iterative_pulse_extraction(
    sig_vs_t, pulse_template, sub_pulse_template,
):
    intermediate_sig_vs_t = []
    sig_vs_t_copy = sig_vs_t.copy()
    arrivalSlices = []
    puls_template_integral = sum(pulse_template)

    iii = 0
    while True:
        sig_conv_sipm = (
            np.convolve(sig_vs_t_copy, pulse_template, mode="valid")
            / puls_template_integral
        )

        if PLOT:
            spe.plot.plot_extraction_state(
                dig=sig_vs_t_copy,
                ADC_FREQUENCY=ADC_FREQUENCY,
                truth=event["true_arrival_times"],
                path="spe_{:06d}.jpg".format(iii),
                ylim=[-1, 5],
            )

        max_slice = int(np.round(np.argmax(sig_conv_sipm) - offset_slices))
        max_response = np.max(sig_conv_sipm)

        if max_response >= 0.32:
            sig_vs_t_copy = spe.add_first_to_second_at(
                f1=sub_pulse_template,
                f2=sig_vs_t_copy,
                injection_slices=[max_slice],
            )
            if max_slice > 0:
                arrivalSlices.append(max_slice + offset_slices * 0.95)
        else:
            break

        iii += 1

    reco_arrival_slices = np.array(arrivalSlices)
    return reco_arrival_slices


reco_arrival_slices = iterative_pulse_extraction(
    sig_vs_t=sig_vs_t,
    pulse_template=pulse_template,
    sub_pulse_template=sub_pulse_template,
)

rr = spe.benchmark(
    arrivalsExtracted=reco_arrival_slices * ADC_CONFIG["skips"],
    arrivalsTruth=event["true_arrival_times"]
    / event["config"]["analog"]["periode"],
    windowRadius=100,
)

event["reco_arrival_times"] = (
    reco_arrival_slices
    * ADC_CONFIG["skips"]
    * event["config"]["analog"]["periode"]
)

spe.plot.plot_event(event=event, path="final")


## FPGA extraction


def timeseries_apply_cooldown(timeseries, cooldown_slices):
    cooldown = np.zeros(len(timeseries), dtype=np.int)
    out = np.zeros(len(timeseries), dtype=np.int)
    for s in range(len(out)):
        if s == 0:
            continue

        if timeseries[s] == 1:
            if cooldown[s - 1] == 0:
                out[s] = 1
                cooldown[s] = cooldown_slices
            else:
                if cooldown[s - 1] > 1:
                    cooldown[s] = cooldown[s - 1] - 1
        else:
            if cooldown[s - 1] > 1:
                cooldown[s] = cooldown[s - 1] - 1

    return out


def fpga_single_extraction_stage(
    sig_vs_t,
    min_baseline_amplitude,
    pulse_template,
    sub_pulse_template,
    sub_offset_slices,
    grad_threshold,
    cooldown_slices,
):
    sig_vs_t_copy = sig_vs_t.copy()
    pulse_template_integral = sum(pulse_template)

    sig_conv_pulse = (
        np.convolve(sig_vs_t_copy, pulse_template, mode="same")
        / pulse_template_integral
    )

    grad_sig_conv_pulse = np.gradient(sig_conv_pulse)
    extraction_candidates = (grad_sig_conv_pulse > grad_threshold).astype(
        np.int
    )

    extraction = timeseries_apply_cooldown(
        timeseries=extraction_candidates, cooldown_slices=cooldown_slices
    )

    # amlitude must not be too low
    valid_baseline = sig_vs_t_copy > min_baseline_amplitude
    extraction = np.logical_and(extraction, valid_baseline)

    extraction_slices = np.where(extraction)[0] - sub_offset_slices

    # subtract
    sig_vs_t_copy = spe.add_first_to_second_at(
        f1=sub_pulse_template,
        f2=sig_vs_t_copy,
        injection_slices=extraction_slices,
    )

    return sig_vs_t_copy.copy(), extraction_slices


COOLDOWN_SLICES = len(sub_pulse_template)
MIN_BASELINE_AMPLITUDE = -0.1 * PULSE_CONFIG["amplitude"]
OFFSET_SLICES = 10 * FPGA_CONFIG["adc_repeats"]
GRAD_THRESHOLDS = [0.2, 0.15, 0.1]


FPGA_sig_vs_t = spe.to_analog_level(
    digital=event["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
)
FPGA_sig_vs_t[0:OFFSET_SLICES] = 0.0

FPGA_pulse_template = spe.to_analog_level(
    digital=ptemp["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
)[3 * FPGA_CONFIG["adc_repeats"] : 11 * FPGA_CONFIG["adc_repeats"]]
spe.plot.plot_extraction_state(
    dig=FPGA_pulse_template,
    ADC_FREQUENCY=FPGA_FREQUENCY,
    truth=event["true_arrival_times"],
    path="FPGA_pulse_template.jpg",
)


sub_pulse_template = -1.0 * spe.to_analog_level(
    digital=ptemp["fpga"],
    amplitude_min=ADC_CONFIG["amplitude_min"],
    amplitude_max=ADC_CONFIG["amplitude_max"],
)
spe.plot.plot_extraction_state(
    dig=sub_pulse_template,
    ADC_FREQUENCY=FPGA_FREQUENCY,
    truth=event["true_arrival_times"],
    path="FPGA_spe_sub_pulse_template.jpg",
)

recos = []
adcs = []

remain_sig_vs_t = sig_vs_t.copy()

for stage in range(len(GRAD_THRESHOLDS)):
    remain_sig_vs_t, reco_arrival_slices = fpga_single_extraction_stage(
        sig_vs_t=remain_sig_vs_t,
        min_baseline_amplitude=MIN_BASELINE_AMPLITUDE,
        pulse_template=pulse_template,
        sub_pulse_template=sub_pulse_template,
        sub_offset_slices=OFFSET_SLICES,
        grad_threshold=GRAD_THRESHOLDS[stage],
        cooldown_slices=COOLDOWN_SLICES,
    )
    adcs.append(remain_sig_vs_t)
    recos.append(reco_arrival_slices)


fig = splt.figure(splt.FIGURE_16_9)
ax = splt.add_axes(fig, [0.1, 0.1, 0.8, 0.8])

ax.plot(sig_vs_t, "k", alpha=0.2)


ax.plot(adcs[0], "r", alpha=0.2)
ax.plot(adcs[1], "g", alpha=0.2)
ax.plot(adcs[2], "b", alpha=0.2)
# ax.plot(adcs[3], "orange", alpha=0.2)


ax.set_xlabel("time / samples")
ax.set_ylabel("amplitude")
fig.savefig("fpga_iteration.jpg")
splt.close_figure(fig)
