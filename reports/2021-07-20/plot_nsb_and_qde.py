import numpy as np
import json
import sebastians_matplotlib_addons as splt
import photon_spectra as phsp

wvl_lim = np.array([200e-9, 1000e-9])
wavelength = np.linspace(wvl_lim[0], wvl_lim[1], 799)
wavelength_bin_width = np.gradient(wavelength)

nsb_hofmann = phsp.nsb_la_palma_2002_hofmann.differential_flux[
    "wavelength_vs_value"
]

cer_hist = phsp.cherenkov_chile.intensity["wavelength_vs_value"]
cer_hist.insert(0, [200e-9, 0.0])
cer_hist.append([850e-9, 38])
cer_hist.append([1000e-9, 22])
cer_hist = np.array(cer_hist)
cer_hist[:, 1] = cer_hist[:, 1] / np.sum(cer_hist[:, 1])
cer = cer_hist

nsb = phsp.nsb_la_palma_2013_benn.differential_flux["wavelength_vs_value"]
# extrapolate
nsb.insert(0, [200e-9, 0.5 * nsb[0][1]])
nsb.append([1000e-9, 2 * nsb[-1][1]])

pmt = phsp.hamamatsu_r11920_100_05.efficiency["wavelength_vs_value"]
pmt.append([1000e-9, 0.0])

sipm = phsp.hamamatsu_s10362_33_050c.efficiency["wavelength_vs_value"]
# extrapolate
sipm.insert(0, [200e-9, 0])
sipm.append([1000e-9, 0.0])

sipm = np.array(sipm)
pmt = np.array(pmt)
cer = np.array(cer)
nsb = np.array(nsb)
nsb_hofmann = np.array(nsb_hofmann)

# assert all range is covered
assert sipm[0, 0] == wvl_lim[0]
assert pmt[0, 0] == wvl_lim[0]
assert nsb[0, 0] == wvl_lim[0]

assert sipm[0, 1] == 0
assert pmt[0, 1] == 0

assert sipm[-1, 0] == wvl_lim[1]
assert pmt[-1, 0] == wvl_lim[1]
assert nsb[-1, 0] == wvl_lim[1]

assert sipm[-1, 1] == 0
assert pmt[-1, 1] == 0

figstyle = {"rows": 720, "cols": 1440, "fontsize": 1}
axspan = [0.15, 0.15, 0.8, 0.8]

# plot NSB
# --------
fig = splt.figure(figstyle)
ax = splt.add_axes(fig, axspan)
ax.plot(
    nsb[:, 0] * 1e9, nsb[:, 1], "k",
)
ax.plot(
    nsb_hofmann[:, 0] * 1e9, nsb_hofmann[:, 1], "k:",
)
ax.plot(
    cer[:, 0] * 1e9, cer[:, 1] * 3e21, "k", alpha=0.24,
)
ax.semilogy()
ax.set_ylabel("diff. flux / m$^{-2}$ sr$^{-1}$ s$^{-1}$ m$^{-1}$")
ax.set_xlabel("wavelength / nm")
ax.set_xlim(wvl_lim * 1e9)
fig.savefig("night_sky_background.jpg")
splt.close_figure(fig)

# plot pde
# --------
fig = splt.figure(figstyle)
ax = splt.add_axes(fig, axspan)
ax.plot(
    pmt[:, 0] * 1e9, pmt[:, 1], "k:",
)
ax.plot(
    sipm[:, 0] * 1e9, sipm[:, 1], "k--",
)
ax.set_ylabel("efficiency / 1")
ax.set_xlabel("wavelength / nm")
ax.set_xlim(wvl_lim * 1e9)
fig.savefig("photon_efficiency.jpg")
splt.close_figure(fig)


# filters and mirrors with coatings
# ---------------------------------

nsb_filter = phsp.veritas_nsb_filter_2015.transmission["wavelength_vs_value"]
nsb_filter.insert(0, [200e-9, 0.0])
nsb_filter.append([1000e-9, 0.0])

mst_mirror = phsp.cta_mirrors.reflectivities["cta_mst_dielectric_after"][
    "wavelength_vs_value"
]
mst_mirror.insert(0, [200e-9, 0.0])
mst_mirror.append([720e-9, 0.0])
mst_mirror.append([1000e-9, 0.0])

mst_mirror = np.array(mst_mirror)
nsb_filter = np.array(nsb_filter)

# assert all range is covered
assert mst_mirror[0, 0] == wvl_lim[0]
assert nsb_filter[0, 0] == wvl_lim[0]

assert mst_mirror[0, 1] == 0
assert nsb_filter[0, 1] == 0

assert mst_mirror[-1, 0] == wvl_lim[1]
assert nsb_filter[-1, 0] == wvl_lim[1]

assert mst_mirror[-1, 1] == 0
assert nsb_filter[-1, 1] == 0


fig = splt.figure(figstyle)
ax = splt.add_axes(fig, axspan)
ax.plot(
    nsb_filter[:, 0] * 1e9, nsb_filter[:, 1], "k:",
)
ax.plot(
    mst_mirror[:, 0] * 1e9, mst_mirror[:, 1], "k--",
)
ax.set_ylabel("transmissivity, reflectivity / 1")
ax.set_xlabel("wavelength / nm")
ax.set_xlim(wvl_lim * 1e9)
fig.savefig("nsb_filters.jpg")
splt.close_figure(fig)



# compute pde-nsb
# ---------------
eq_nsb = phsp._to_array_interp(wavelength_vs_value=nsb, wavelengths=wavelength)
eq_pmt = phsp._to_array_interp(wavelength_vs_value=pmt, wavelengths=wavelength)
eq_sipm = phsp._to_array_interp(
    wavelength_vs_value=sipm, wavelengths=wavelength
)
nsb_pmt = eq_nsb * eq_pmt
nsb_sipm = eq_nsb * eq_sipm
I_nsb_pmt = np.sum(nsb_pmt * wavelength_bin_width)
I_nsb_sipm = np.sum(nsb_sipm * wavelength_bin_width)

sqdeg2sr = np.deg2rad(1) ** 2

eq_nsb_filter = phsp._to_array_interp(
    wavelength_vs_value=nsb_filter, wavelengths=wavelength
)
eq_mst_mirror = phsp._to_array_interp(
    wavelength_vs_value=mst_mirror, wavelengths=wavelength
)

nsb_pmt_mirror = eq_nsb * eq_pmt * eq_mst_mirror
nsb_sipm_mirror = eq_nsb * eq_sipm * eq_mst_mirror

I_nsb_pmt_mirror = np.sum(nsb_pmt_mirror * wavelength_bin_width)
I_nsb_sipm_mirror= np.sum(nsb_sipm_mirror * wavelength_bin_width)

nsb_pmt_mirror_filter = eq_nsb * eq_pmt * eq_mst_mirror * eq_nsb_filter
nsb_sipm_mirror_filter = eq_nsb * eq_sipm * eq_mst_mirror * eq_nsb_filter

I_nsb_pmt_mirror_filter = np.sum(nsb_pmt_mirror_filter * wavelength_bin_width)
I_nsb_sipm_mirror_filter = np.sum(nsb_sipm_mirror_filter * wavelength_bin_width)


eq_cer = phsp._to_array_interp(
    wavelength_vs_value=cer, wavelengths=wavelength
)
# normalize so integral is 1.0
eq_cer = eq_cer / np.sum(eq_cer)
eq_cer = eq_cer / wavelength_bin_width




nsb_table = {
    "sipm_________M_per_m2_per_deg2_per_s": I_nsb_sipm * sqdeg2sr * 1e-6,
    "sipm_mir_____M_per_m2_per_deg2_per_s": I_nsb_sipm_mirror * sqdeg2sr * 1e-6,
    "sipm_mir_fil_M_per_m2_per_deg2_per_s": I_nsb_sipm_mirror_filter * sqdeg2sr * 1e-6,
    "pmt__________M_per_m2_per_deg2_per_s": I_nsb_pmt * sqdeg2sr * 1e-6,
    "pmt__mir_____M_per_m2_per_deg2_per_s": I_nsb_pmt_mirror * sqdeg2sr * 1e-6,
    "pmt__mir_fil_M_per_m2_per_deg2_per_s": I_nsb_pmt_mirror_filter * sqdeg2sr * 1e-6,
}

wbw = wavelength_bin_width
cer_table = {
    "sipm_________": np.sum(eq_sipm * eq_cer * wbw),
    "sipm_mir_____": np.sum(eq_sipm * eq_cer * eq_mst_mirror * wbw),
    "sipm_mir_fil_": np.sum(eq_sipm * eq_cer * eq_mst_mirror * eq_nsb_filter * wbw),
    "pmt__________": np.sum(eq_pmt * eq_cer * wbw),
    "pmt__mir_____": np.sum(eq_pmt * eq_cer * eq_mst_mirror * wbw),
    "pmt__mir_fil_": np.sum(eq_pmt * eq_cer * eq_mst_mirror * eq_nsb_filter * wbw),
}

with open("nsb_rates_in_sensors.json", "wt") as f:
    f.write(json.dumps({"nsb": nsb_table, "cer": cer_table}, indent=4))

