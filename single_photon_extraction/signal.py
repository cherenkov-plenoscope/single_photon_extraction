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


def make_analog_output(periode, perfect, lowpass_cutoff_frequency):
    scaling_to_let_the_gain_go_down_to_one_half_at_cutoff_frequency = 0.19
    bandwitdh_kernel = make_bell(
        periode=periode,
        bell_std=(
            scaling_to_let_the_gain_go_down_to_one_half_at_cutoff_frequency /
            lowpass_cutoff_frequency
        )
    )
    analog = np.convolve(perfect, bandwitdh_kernel, mode="same")
    return analog


def make_adc_output(
    analog, skips, amplitude_noise, amplitude_min, amplitude_max, num_bits, prng,
):
    sample_slices = np.arange(0, len(analog), skips)
    out = analog[sample_slices]

    # add noise
    assert amplitude_noise >= 0.0
    noise = prng.normal(loc=0.0, scale=amplitude_noise, size=(len(out)))
    out += noise

    # clipping
    assert amplitude_max > amplitude_min
    out[out >= amplitude_max] = amplitude_max
    out[out <= amplitude_min] = amplitude_min

    # normalize
    out -= amplitude_min
    out = out / (amplitude_max - amplitude_min)

    # extend to bit range
    out = out * (2**num_bits)
    return out.astype(np.int)


def make_fpga_output(
    adc, fpga_adc_repeats, fpga_kernel,
    fpga_num_bits, adc_num_bits
):
    fpga = np.repeat(adc, repeats=fpga_adc_repeats)
    if len(fpga_kernel) > 0:
        fpga = np.convolve(fpga, fpga_kernel, mode="same")
        fpga = fpga / np.sum(fpga_kernel)

    # bit range
    fpga = fpga * (2**(fpga_num_bits - adc_num_bits))
    return fpga


def to_analog_level(
    digital, amplitude_min, amplitude_max, num_bits
):
    ana = digital.astype(np.float)
    ana /= (2**num_bits - 1)
    ana *= amplitude_max - amplitude_min
    ana += amplitude_min
    return ana


def power_spectrum(periode, sig_vs_t):
    spec = np.fft.fft(sig_vs_t)
    freq = np.fft.fftfreq(spec.size, d=periode)
    idx = np.argsort(freq)
    freq = freq[idx]
    spec = spec[idx]
    spec = np.sqrt(np.real(spec)**2.0 + np.imag(spec)**2.0)
    return freq, spec
