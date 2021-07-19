import numpy as np


def make_timeseries(num_samples, periode, time_start=0.0):
    return np.linspace(
        time_start,
        time_start + num_samples * periode,
        num_samples,
        endpoint=False,
    )


def make_pulse(periode, pulse_amplitude, pulse_decay_time, num_samples=None):
    if num_samples is None:
        num_samples = 1
        while True:
            t = num_samples * periode
            amp = np.exp(-t / pulse_decay_time)
            if amp <= 0.01:
                break
            else:
                num_samples *= 2

    ts = make_timeseries(
        num_samples=num_samples, periode=periode, time_start=0.0
    )
    amp = pulse_amplitude * np.exp(-ts / pulse_decay_time)
    assert amp[-1] <= (0.01 * pulse_amplitude)
    return amp


def make_bell(periode, bell_std, num_samples=None):
    if num_samples is None:
        num_samples = 1
        while True:
            t = 0.5 * num_samples * periode
            amp = np.exp(-0.5 * (t / bell_std) ** 2)
            if amp <= 0.01:
                break
            else:
                num_samples *= 2

    exposure = num_samples * periode
    ts = make_timeseries(
        num_samples=num_samples, periode=periode, time_start=-exposure / 2
    )
    b = np.exp(-0.5 * (ts / bell_std) ** 2)
    assert b[0] <= 0.01
    return b / np.sum(b)


def add_first_to_second_at(f1, f2, injection_slices):
    """
    Adds the first 1D array to the second 1D array at all injection
    slices of the third 1D array.
    """
    out = np.array(f2)
    for injection_slice in injection_slices:
        # injection point exceeds range of f2
        if injection_slice > f2.shape[0]:
            continue

        if injection_slice < 0:
            injection_slice = 0

        # endpoint of injection in f2: e2
        e2 = injection_slice + f1.shape[0]

        # naive endpoint of sampling in f1
        e1 = f1.shape[0]

        # e2 is limited to range of f2
        if e2 > f2.shape[0]:
            e2 = f2.shape[0]

        # correct sampling range in f1 if needed
        if e2 - injection_slice < f1.shape[0]:
            e1 = e2 - injection_slice

        out[injection_slice:e2] += f1[0:e1]
    return out


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


def make_adc_output(
    analog, skips, noise_amplitude, amplitude_min, amplitude_max, prng,
):
    sample_slices = np.arange(0, len(analog), skips)
    out = analog[sample_slices]

    # add noise
    assert noise_amplitude >= 0.0
    noise = prng.normal(loc=0.0, scale=noise_amplitude, size=(len(out)))
    out += noise

    # clipping
    assert amplitude_max > amplitude_min
    out[out >= amplitude_max] = amplitude_max
    out[out <= amplitude_min] = amplitude_min

    # normalize
    out -= amplitude_min
    out = out / (amplitude_max - amplitude_min)

    # extend to bit range
    out = out * 256
    return out.astype(np.uint8)


def make_fpga_output(
    adc, fpga_adc_repeats, fpga_kernel,
):
    fpga = np.repeat(adc, repeats=fpga_adc_repeats)
    if len(fpga_kernel) > 0:
        fpga = np.convolve(fpga, fpga_kernel, mode="same")
        fpga = fpga / np.sum(fpga_kernel)
    return fpga


def to_analog_level(
    digital, amplitude_min, amplitude_max,
):
    ana = digital.astype(np.float)
    ana /= 255
    ana *= amplitude_max - amplitude_min
    ana += amplitude_min
    return ana


def benchmark(arrivalsExtracted, arrivalsTruth, windowRadius=10):
    arrivalsExtracted = np.sort(arrivalsExtracted)
    arrivalsTruth = np.sort(arrivalsTruth)

    def find_nearest(array, value):
        return (np.abs(array - value)).argmin()

    bench = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    arrivalsExtractedRemaining = arrivalsExtracted.copy()
    for arrivalTruth in arrivalsTruth:
        if arrivalsExtractedRemaining.shape[0] == 0:
            bench["fn"] += 1
        else:
            match = find_nearest(arrivalsExtractedRemaining, arrivalTruth)
            distance = np.abs(arrivalsExtractedRemaining[match] - arrivalTruth)
            if distance <= windowRadius:
                arrivalsExtractedRemaining = np.delete(
                    arrivalsExtractedRemaining, match
                )
                bench["tp"] += 1
            else:
                bench["fn"] += 1
    bench["fp"] += arrivalsExtractedRemaining.shape[0]
    return bench


def make_night_sky_background_event(
    num_samples,
    analog_periode,
    analog_bandwidth,
    pulse_config,
    adc_config,
    fpga_config,
    nsb_rate,
    prng,
):
    assert num_samples > 0
    assert analog_periode > 0
    assert analog_bandwidth > 0

    assert pulse_config["amplitude_std"] >= 0
    assert pulse_config["decay_time"] >= 0

    assert nsb_rate >= 0

    exposure_time = num_samples * analog_periode

    true_arrival_times = draw_poisson_arrival_times(
        exposure_time=exposure_time, frequency=nsb_rate, prng=prng,
    )
    true_arrival_slices = (true_arrival_times / analog_periode).astype(np.int)
    true_arrival_times = analog_periode * true_arrival_slices

    analog = make_timeseries(num_samples=num_samples, periode=analog_periode)

    for true_arrival_slice in true_arrival_slices:
        amp = pulse_config["amplitude"] + prng.normal(
            loc=0, scale=pulse_config["amplitude_std"]
        )

        p = make_pulse(
            num_samples=None,
            periode=analog_periode,
            pulse_amplitude=amp,
            pulse_decay_time=pulse_config["decay_time"],
        )

        analog = add_first_to_second_at(
            f1=p, f2=analog, injection_slices=[true_arrival_slice],
        )

    bandwidth_periode = 1 / analog_bandwidth

    bandwidth_kernel = make_bell(
        num_samples=None, periode=analog_periode, bell_std=bandwidth_periode
    )

    analog = np.convolve(analog, bandwidth_kernel, mode="same")

    adc = make_adc_output(
        analog=analog,
        skips=adc_config["skips"],
        noise_amplitude=adc_config["noise"],
        amplitude_min=adc_config["amplitude_min"],
        amplitude_max=adc_config["amplitude_max"],
        prng=prng,
    )

    fpga = make_fpga_output(
        adc=adc,
        fpga_adc_repeats=fpga_config["adc_repeats"],
        fpga_kernel=fpga_config["kernel"],
    )

    return {
        "true_arrival_times": true_arrival_times,
        "analog": analog,
        "adc": adc,
        "fpga": fpga,
        "config": {
            "analog": {
                "periode": analog_periode,
                "bandwidth": analog_bandwidth,
            },
            "adc": adc_config,
            "fpga": fpga_config,
        },
    }


def make_pulse_template(
    periode, analog_config, pulse_config, adc_config, fpga_config, prng,
):
    pulse = make_pulse(
        periode=periode,
        pulse_amplitude=pulse_config["amplitude"],
        pulse_decay_time=pulse_config["decay_time"],
    )
    bandwitdh_kernel = make_bell(
        periode=periode, bell_std=(1 / analog_config["bandwidth"])
    )

    analog = make_timeseries(num_samples=len(pulse) + 200, periode=periode)
    analog = add_first_to_second_at(
        f1=pulse, f2=analog, injection_slices=[200],
    )
    analog = np.convolve(analog, bandwitdh_kernel, mode="same")

    adc = make_adc_output(
        analog=analog,
        skips=adc_config["skips"],
        noise_amplitude=0.0,
        amplitude_min=adc_config["amplitude_min"],
        amplitude_max=adc_config["amplitude_max"],
        prng=prng,
    )

    fpga = make_fpga_output(
        adc=adc,
        fpga_adc_repeats=fpga_config["adc_repeats"],
        fpga_kernel=fpga_config["kernel"],
    )

    return {
        "analog": analog,
        "adc": adc,
        "fpga": fpga,
        "config": {
            "analog": {"periode": periode},
            "adc": adc_config,
            "fpga": fpga_config,
        },
    }
