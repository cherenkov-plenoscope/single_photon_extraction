from . import signal
from . import extractors
import numpy as np


def draw_poisson_arrival_times(exposure_time, frequency, prng):
    """
    poisson distributed arrival times of pulses in a given exposure time
    exposure_time with an average frequency of f
    """
    arrival_times = []
    time = 0
    while time < exposure_time:
        time_until_next_arrival = -np.log(prng.uniform()) / frequency
        time += time_until_next_arrival
        if time < exposure_time:
            arrival_times.append(time)
    return np.array(arrival_times)


def benchmark(reco_times, true_times, time_delta):
    reco_times = np.sort(reco_times)
    true_times = np.sort(true_times)

    def find_nearest(array, value):
        return (np.abs(array - value)).argmin()

    bench = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    reco_times_remaining = reco_times.copy()
    for true_time in true_times:
        if reco_times_remaining.shape[0] == 0:
            bench["fn"] += 1
        else:
            match = find_nearest(reco_times_remaining, true_time)
            distance = np.abs(reco_times_remaining[match] - true_time)
            if distance <= time_delta:
                reco_times_remaining = np.delete(reco_times_remaining, match)
                bench["tp"] += 1
            else:
                bench["fn"] += 1
    bench["fp"] += reco_times_remaining.shape[0]
    return bench


def analyse_benchmark(tp, tn, fp, fn):
    true_positive_rate = tp / (tp + fn)
    D_tp = np.sqrt(tp)
    D_fn = np.sqrt(fn)
    dS_dtp = fn / (tp + fn) ** 2
    dS_dfn = -tp / (tp + fn) ** 2
    true_positive_rate_unc = np.sqrt(
        (dS_dtp ** 2) * (D_tp ** 2) + (dS_dfn ** 2) * (D_fn ** 2)
    )
    false_negative_rate = fp / (tp + fn)
    D_fp = np.sqrt(fp)
    D_tp = np.sqrt(tp)
    D_fn = np.sqrt(fn)

    dM_dfp = 1 / (tp + fn)
    dM_dtp = -fp / (tp + fn) ** 2
    dM_dfn = -fp / (tp + fn) ** 2
    false_negative_rate_unc = np.sqrt(
        (dM_dfp ** 2) * (D_fp ** 2)
        + (dM_dtp ** 2) * (D_tp ** 2)
        + (dM_dfn ** 2) * (D_fn ** 2)
    )
    return {
        "true_positive_rate_mean": true_positive_rate,
        "true_positive_rate_uncertainty": true_positive_rate_unc,
        "false_negative_rate_mean": false_negative_rate,
        "false_negative_rate_uncertainty": false_negative_rate_unc,
    }


def make_night_sky_background_event(
    num_samples,
    analog_config,
    pulse_config,
    adc_config,
    fpga_config,
    nsb_rate,
    prng,
):
    assert num_samples > 0
    assert analog_config["periode"] > 0
    assert analog_config["bandwidth"] > 0

    assert pulse_config["amplitude_std"] >= 0
    assert pulse_config["decay_time"] >= 0

    assert nsb_rate >= 0

    exposure_time = num_samples * analog_config["periode"]

    true_arrival_times = draw_poisson_arrival_times(
        exposure_time=exposure_time, frequency=nsb_rate, prng=prng,
    )
    true_arrival_slices = (
        true_arrival_times / analog_config["periode"]
    ).astype(np.int)
    true_arrival_times = analog_config["periode"] * true_arrival_slices

    perfect = np.zeros(num_samples)

    for true_arrival_slice in true_arrival_slices:
        amp = prng.normal(
            loc=pulse_config["amplitude"], scale=pulse_config["amplitude_std"]
        )

        dec = prng.normal(
            loc=pulse_config["decay_time"],
            scale=pulse_config["decay_time_std"],
        )

        p = signal.make_pulse(
            num_samples=None,
            periode=analog_config["periode"],
            pulse_amplitude=amp,
            pulse_decay_time=dec,
        )

        perfect = signal.add_first_to_second_at(
            f1=p, f2=perfect, injection_slices=[true_arrival_slice],
        )

    analog = signal.make_analog_output(
        periode=analog_config["periode"],
        perfect=perfect,
        lowpass_cutoff_frequency=analog_config["bandwidth"],
    )

    adc = signal.make_adc_output(
        analog=analog,
        skips=adc_config["skips"],
        amplitude_noise=adc_config["amplitude_noise"],
        amplitude_min=adc_config["amplitude_min"],
        amplitude_max=adc_config["amplitude_max"],
        num_bits=adc_config["num_bits"],
        prng=prng,
    )

    fpga = signal.make_fpga_output(
        adc=adc,
        fpga_adc_repeats=fpga_config["adc_repeats"],
        fpga_kernel=fpga_config["kernel"],
        fpga_num_bits=fpga_config["num_bits"],
        adc_num_bits=adc_config["num_bits"],
    )

    return {
        "true_arrival_times": true_arrival_times,
        "analog": analog,
        "adc": adc,
        "fpga": fpga,
        "config": {
            "analog": analog_config,
            "adc": adc_config,
            "fpga": fpga_config,
        },
    }
