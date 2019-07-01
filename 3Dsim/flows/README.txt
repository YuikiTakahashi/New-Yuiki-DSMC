New labelling/organization for DSMC flow field files. 

We label each flow field by a Geometry Type (f, g, h, ...) which denotes the type of geometry implemented (hole, bevel, de Laval).
Additionally, a three digit number labels the input buffer gas flowrate in SCCM.
Every DSMC flow field filename will begin with the string 'DS2'.
Directory-wise, the flowfields are organized into folders according to geometry.

Thus, F_Cell/DS2f050.DAT denotes the field with geometry 'f' and flowrate=50 SCCM.
