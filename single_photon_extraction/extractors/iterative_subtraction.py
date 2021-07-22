import numpy as np
from .. import signal


def timeseries_apply_cooldown(timeseries, num_cooldown_slices):
    cooldown = np.zeros(len(timeseries), dtype=np.int)
    out = np.zeros(len(timeseries), dtype=np.int)
    for s in range(len(out)):
        if s == 0:
            continue

        if timeseries[s] == 1:
            if cooldown[s - 1] == 0 and timeseries[s - 1] == 0:
                out[s] = 1
                cooldown[s] = num_cooldown_slices
            else:
                if cooldown[s - 1] > 1:
                    cooldown[s] = cooldown[s - 1] - 1
        else:
            if cooldown[s - 1] > 1:
                cooldown[s] = cooldown[s - 1] - 1

    return cooldown, out


def one_stage(
    sig_vs_t,
    min_amplitude_to_subtract_from,
    pulse_rising_edge_template,
    subtraction_pulse_template,
    num_subtraction_offset_slices,
    threshold,
    num_cooldown_slices,
):
    DEBUG = {}
    sig = sig_vs_t.copy()
    DEBUG["sig"] = sig
    pulse_template_integral = sum(pulse_rising_edge_template)

    sig_conv_pulse = (
        np.convolve(sig, pulse_rising_edge_template, mode="same")
        / pulse_template_integral
    )
    DEBUG["sig_conv_pulse"] = sig_conv_pulse

    extraction_candidates = (sig_conv_pulse > threshold).astype(np.int)
    DEBUG["extraction_candidates"] = extraction_candidates

    cooldown, extraction = timeseries_apply_cooldown(
        timeseries=extraction_candidates,
        num_cooldown_slices=num_cooldown_slices,
    )
    DEBUG["cooldown"] = cooldown

    # amlitude must not be too low
    valid_baseline = sig > min_amplitude_to_subtract_from
    DEBUG["valid_baseline"] = valid_baseline

    extraction = np.logical_and(extraction, valid_baseline)
    DEBUG["extraction"] = extraction

    extraction_slices = np.where(extraction)[0] - num_subtraction_offset_slices

    # subtract
    sig = signal.add_first_to_second_at(
        f1=subtraction_pulse_template,
        f2=sig,
        injection_slices=extraction_slices,
    )

    return sig, extraction_slices, DEBUG


def apply(sampling_periode, sig, config):
    DEBUG = {}
    DEBUG["stage_debugs"] = []
    DEBUG["stage_reco_arrival_slices"] = []

    num_stages = len(config["stage_thresholds"])
    remain_sig = sig.copy()
    for stage in range(num_stages):
        next_sig, stage_reco_arrival_slices, stage_debug = one_stage(
            sig_vs_t=remain_sig,
            min_amplitude_to_subtract_from=config[
                "min_amplitude_to_subtract_from"
            ],
            pulse_rising_edge_template=config["pulse_rising_edge_template"],
            subtraction_pulse_template=config["subtraction_pulse_template"],
            num_subtraction_offset_slices=config[
                "num_subtraction_offset_slices"
            ],
            threshold=config["stage_thresholds"][stage],
            num_cooldown_slices=config["num_cooldown_slices"],
        )

        DEBUG["stage_debugs"].append(stage_debug)
        DEBUG["stage_reco_arrival_slices"].append(stage_reco_arrival_slices)

        remain_sig = next_sig

    all_reco_arrival_slices = np.concatenate(
        DEBUG["stage_reco_arrival_slices"]
    )
    all_reco_arrival_slices += int(
        0.5 * config["num_subtraction_offset_slices"]
    )

    all_reco_arrival_times = all_reco_arrival_slices * sampling_periode
    all_reco_arrival_times = np.sort(all_reco_arrival_times)

    return all_reco_arrival_times, DEBUG
