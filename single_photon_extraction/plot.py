from single_photon_extraction import *
import numpy as np
import sebastians_matplotlib_addons as splt



def plot_event(
    event,
    path,
    figstyle=splt.FIGURE_16_9,
    axspan=[0.1, 0.1, 0.8, 0.8],
    ylim=None,
):
    analog_timeseries = make_timeseries(
        num_samples=len(event["analog"]),
        periode=event["config"]["analog"]["periode"],
    )
    adc_timeseries = make_timeseries(
        num_samples=len(event["adc"]),
        periode=(
            event["config"]["analog"]["periode"]
            * event["config"]["adc"]["skips"]
        ),
    )

    fig = splt.figure(figstyle)
    ax = splt.add_axes(fig, axspan)

    # analog
    # ------
    ax.plot(analog_timeseries, event["analog"], "k")

    # adc
    # ---
    splt.ax_add_histogram(
        ax=ax,
        bin_edges=adc_timeseries,
        bincounts=to_analog_level(
            digital=event["adc"][:-1],
            amplitude_min=event["config"]["adc"]["amplitude_min"],
            amplitude_max=event["config"]["adc"]["amplitude_max"],
        ),
        linecolor="red",
        linealpha=0.2,
        draw_bin_walls=True,
    )

    # fpga
    # ----
    if "fpga" in event:
        fpga_timeseries = make_timeseries(
            num_samples=len(event["fpga"]),
            periode=(
                event["config"]["analog"]["periode"]
                * event["config"]["adc"]["skips"]
                / event["config"]["fpga"]["adc_repeats"]
            ),
        )
        ax.plot(
            fpga_timeseries,
            to_analog_level(
                digital=event["fpga"],
                amplitude_min=event["config"]["adc"]["amplitude_min"],
                amplitude_max=event["config"]["adc"]["amplitude_max"],
            ),
            "red",
        )

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

    ax.set_ylim(ylim)
    ax.set_xlabel("time / s")
    ax.set_ylabel("amplitude")
    fig.savefig(path)
    splt.close_figure(fig)


def plot_extraction_state(
    dig, ADC_FREQUENCY, truth, path, ylim=None
):
    periode = 1.0 / ADC_FREQUENCY
    fig = splt.figure(splt.FIGURE_16_9)
    ax = splt.add_axes(fig, [0.1, 0.1, 0.8, 0.8])
    splt.ax_add_histogram(
        ax=ax,
        bin_edges=make_timeseries(len(dig), periode=periode),
        bincounts=dig[:-1],
        linecolor="red",
        draw_bin_walls=True,
    )
    ax.set_ylim(ylim)
    ax.set_xlabel("time / s")
    fig.savefig(path)
    splt.close_figure(fig)
