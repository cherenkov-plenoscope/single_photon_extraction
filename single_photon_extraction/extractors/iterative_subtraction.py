import numpy as np
from .. import signal


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
    debug = {}
    sig_vs_t_copy = sig_vs_t.copy()
    pulse_template_integral = sum(pulse_template)

    sig_conv_pulse = (
        np.convolve(sig_vs_t_copy, pulse_template, mode="same")
        / pulse_template_integral
    )
    debug["sig_conv_pulse"] = sig_conv_pulse

    extraction_candidates = (sig_conv_pulse > grad_threshold).astype(
        np.int
    )
    debug["extraction_candidates"] = extraction_candidates

    extraction = timeseries_apply_cooldown(
        timeseries=extraction_candidates, cooldown_slices=cooldown_slices
    )

    # amlitude must not be too low
    valid_baseline = sig_vs_t_copy > min_baseline_amplitude
    debug["valid_baseline"] = valid_baseline

    extraction = np.logical_and(extraction, valid_baseline)
    debug["extraction"] = extraction

    extraction_slices = np.where(extraction)[0] - sub_offset_slices

    # subtract
    sig_vs_t_copy = signal.add_first_to_second_at(
        f1=sub_pulse_template,
        f2=sig_vs_t_copy,
        injection_slices=extraction_slices,
    )

    return sig_vs_t_copy.copy(), extraction_slices, debug
