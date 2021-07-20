import numpy as np
import json
import sebastians_matplotlib_addons as splt
import photon_spectra as phsp

wvl_lim = np.array([200e-9, 1000e-9])
wavelength = np.linspace(wvl_lim[0], wvl_lim[1], 799)


nsb_hofmann = phsp.nsb_la_palma_2002_hofmann.differential_flux["wavelength_vs_value"]

nsb = phsp.nsb_la_palma_2013_benn.differential_flux["wavelength_vs_value"]
# extrapolate
nsb.insert(0, [200e-9, 0.5*nsb[0][1]])
nsb.append([1000e-9, 2*nsb[-1][1]])

pmt = phsp.hamamatsu_r11920_100_05.efficiency["wavelength_vs_value"]
pmt.append([1000e-9, 0.0])

sipm = phsp.hamamatsu_s10362_33_050c.efficiency["wavelength_vs_value"]
# extrapolate
sipm.insert(0, [200e-9, 0])
sipm.append([1000e-9, 0.0])

sipm = np.array(sipm)
pmt = np.array(pmt)
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


# plot NSB
# --------
fig = splt.figure(splt.FIGURE_16_9)
ax = splt.add_axes(fig, [0.15, 0.1, 0.8, 0.8])
ax.plot(
    nsb[:, 0]*1e9,
    nsb[:, 1],
    "k",
)
ax.plot(
    nsb_hofmann[:, 0]*1e9,
    nsb_hofmann[:, 1],
    "k:",
)
ax.semilogy()
ax.set_ylabel("diff. flux / m$^{-2}$ sr$^{-1}$ s$^{-1}$ m$^{-1}$")
ax.set_xlabel("wavelength / nm")
ax.set_xlim(wvl_lim*1e9)
fig.savefig("night_sky_background.jpg")
splt.close_figure(fig)

# plot pde
# --------
fig = splt.figure(splt.FIGURE_16_9)
ax = splt.add_axes(fig, [0.15, 0.1, 0.8, 0.8])
ax.plot(
    pmt[:, 0]*1e9,
    pmt[:, 1],
    "k:",
)
ax.plot(
    sipm[:, 0]*1e9,
    sipm[:, 1],
    "k--",
)
ax.set_ylabel("efficiency / 1")
ax.set_xlabel("wavelength / nm")
ax.set_xlim(wvl_lim*1e9)
fig.savefig("photon_efficiency.jpg")
splt.close_figure(fig)


# compute pde-nsb
# ---------------
eq_nsb = phsp._to_array_interp(wavelength_vs_value=nsb, wavelengths=wavelength)
eq_pmt = phsp._to_array_interp(wavelength_vs_value=pmt, wavelengths=wavelength)
eq_sipm = phsp._to_array_interp(wavelength_vs_value=sipm, wavelengths=wavelength)
nsb_pmt = eq_nsb * eq_pmt
nsb_sipm = eq_nsb * eq_sipm
wavelength_bin_width = np.gradient(wavelength)
I_nsb_pmt = np.sum(nsb_pmt * wavelength_bin_width)
I_nsb_sipm = np.sum(nsb_sipm * wavelength_bin_width)

sqdeg2sr = np.deg2rad(1)**2

out = {
    "sipm_nsb_rate_M_per_m2_per_sr_per_s": I_nsb_sipm*1e-6,
    "sipm_nsb_rate_M_per_m2_per_deg2_per_s": I_nsb_sipm * sqdeg2sr*1e-6,
    "pmt_nsb_rate_M_per_m2_per_sr_per_s": I_nsb_pmt*1e-6,
    "pmt_nsb_rate_M_per_m2_per_deg2_per_s": I_nsb_pmt * sqdeg2sr*1e-6,
}

with open("nsb_rates_in_sensors.json", "wt") as f:
    f.write(json.dumps(out, indent=4))

