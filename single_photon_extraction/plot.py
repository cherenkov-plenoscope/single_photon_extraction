import numpy as np
import single_photon_extraction as spe
import sebastians_matplotlib_addons as splt


def ax_add_event(ax, event):
    # truth
    # -----
    if "true_arrival_times" in event:
        for true_arrival_time in event["true_arrival_times"]:
            ax.axvline(true_arrival_time, color="b")

    # reco
    # ----
    if "reco_arrival_times" in event:
        for reco_arrival_time in event["reco_arrival_times"]:
            ax.axvline(reco_arrival_time, color="g", linestyle="--")


    # analog
    # ------
    if "analog" in event:
        analog_timeseries = spe.signal.make_timeseries(
            num_samples=len(event["analog"]),
            periode=event["config"]["analog"]["periode"],
        )
        ax.plot(analog_timeseries, event["analog"], "k", alpha=0.5)

    # adc
    # ---
    if "adc" in event:
        adc_timeseries = spe.signal.make_timeseries(
            num_samples=len(event["adc"]),
            periode=(
                event["config"]["analog"]["periode"]
                * event["config"]["adc"]["skips"]
            ),
        )

        splt.ax_add_histogram(
            ax=ax,
            bin_edges=adc_timeseries,
            bincounts=spe.signal.to_analog_level(
                digital=event["adc"][:-1],
                amplitude_min=event["config"]["adc"]["amplitude_min"],
                amplitude_max=event["config"]["adc"]["amplitude_max"],
                num_bits=event["config"]["adc"]["num_bits"],
            ),
            linecolor="orange",
            linealpha=0.5,
            draw_bin_walls=True,
        )

    # fpga
    # ----
    if "fpga" in event:
        fpga_timeseries = spe.signal.make_timeseries(
            num_samples=len(event["fpga"]),
            periode=(
                event["config"]["analog"]["periode"]
                * event["config"]["adc"]["skips"]
                / event["config"]["fpga"]["adc_repeats"]
            ),
        )
        ax.step(
            fpga_timeseries,
            spe.signal.to_analog_level(
                digital=event["fpga"],
                amplitude_min=event["config"]["adc"]["amplitude_min"],
                amplitude_max=event["config"]["adc"]["amplitude_max"],
                num_bits=event["config"]["fpga"]["num_bits"],
            ),
            "red",
        )

    return ax
