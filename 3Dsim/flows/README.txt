New labelling/organization for DSMC flow field files. 

We label each flow field by a Geometry Type (f, g, h, ...) which denotes the type of geometry implemented (hole, bevel, de Laval).
Additionally, a three digit number labels the input buffer gas flowrate in SCCM.
Every DSMC flow field filename will begin with the string 'DS2'.
Directory-wise, the flowfields are organized into folders according to geometry.

Thus, F_Cell/DS2f050.DAT denotes the field with geometry 'f' and flowrate=50 SCCM.

Y Cell; a single stage cell created by Yuiki
T Cell; a single stage cell with a He inlet tube created by Yuiki
X Cell old; a single stage cell with a He inlet tube (and the whole entire geometry is extended) created by Yuiki
U cell; a single stage cell with a He inlet tube (and the whole entire geometry is extended) containing 16K He flow field data
V cell; a single stage cell with a He inlet tube (and the whole entire geometry is extended) containing 16K Ne flow field data
X Cell; This is 4K He flowfield data but re-ran Xcell old DS2V with some tweaks.  "Y geometry w tube - He4K -NRH-20200519"
(In addition to changing the geometry to eliminate regions with no flow (therefore reducing the computational space) I adjusted the cell sizes so that it would converge faster.)

Z cell; X cell old but with thicker aperture (thickness of aperture is changed from 0.5 mm to 2.5 mm)
W Cell; 2 stage cell with 4.7 cm gap, no mesh, no vents
C cell; conical gemoetry, open angle is 106 degrees

If you change the cell geometry, for example, from T Cell to X cell, do not forget to change line 974 in fluidSim3D_Yuiki3.py from /T_Cell/.. to /X_Cell/..