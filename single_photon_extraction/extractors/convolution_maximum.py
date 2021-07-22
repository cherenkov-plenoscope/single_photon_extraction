import numpy as np
from .. import signal


def apply(sampling_periode, sig, config):
    DEBUG = {}
    DEBUG["sig"] = sig

    baseline_is_valid = sig >= config["min_amplitude"]
    DEBUG["baseline_is_valid"] = baseline_is_valid

    all_reco_arrival_times = []
    sig_conv = (
        np.convolve(sig, config["pulse_rising_edge_template"], mode="same")
        / np.sum(config["pulse_rising_edge_template"])
    )
    DEBUG["sig_conv"] = sig_conv

    sig_conv_high = sig_conv > config["convolution_threshold"]
    DEBUG["sig_conv_high"] = sig_conv_high
    sig_conv_rising = np.gradient(sig_conv) > config["convolution_gradient_threshold"]
    DEBUG["sig_conv_rising"] = sig_conv_rising

    candidate = np.logical_and(sig_conv_high, sig_conv_rising).astype(np.int)
    response = np.logical_and(baseline_is_valid, candidate).astype(np.int)

    DEBUG["response"] = response

    response_rising = np.zeros(len(sig), dtype=np.int)
    for s in range(len(sig)):
        if s > 1:
            if response[s] == 1 and response[s - 1] == 0:
                response_rising[s] = 1
    DEBUG["response_rising"] = response_rising

    reco_arrival_slices = np.where(response_rising)[0]
    reco_arrival_slices += config["num_offset_slices"]
    return  reco_arrival_slices * sampling_periode, DEBUG
