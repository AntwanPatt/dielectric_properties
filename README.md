# Dielectric properties

A python class to calculate the dielectric properties of a system using molecular simulation data.

Main quantities of interest:
* static dielectric constant
* auto-correlation function of total dipole moment
* frenquency dependent dielectric function

TO DO:
```
* think about the workflow between getting M, calculating the ACF and then epsilon
because the fitting procedure of the ACF might need an oversight
* check ACF calculation to make it more statistically accurate if possible
* clean integration of ACF
* clean fit function
* look into outputs, maybe write a dedicated function
* let the opportunity to choose the fitting interval of the ACf
* calculate dielectric function from fit
* add fit models
* get M from trajectory
* calculate Fourier transform of ACF
* calculate dielectric function from Fourier transform
```
