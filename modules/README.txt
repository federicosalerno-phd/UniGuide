UniGuide — unified guide modules
=================================

Drop the STL files of your standardized, reusable guide elements in THIS folder.
They appear automatically in the "Guides" step (no app restart needed — they are
listed each time you enter that step).

Modelling convention (so a module locks onto a cutting plane)
-------------------------------------------------------------
Model each element so that:

  * its CUTTING FACE lies on the local XY plane  (Z = 0),
  * the face NORMAL points along +Z,
  * the part is roughly CENTRED on X and Y (symmetric about the local origin).

The app places that local origin on the chosen cutting plane, keeping the face
coincident with the plane, and lets you slide it in-plane (X/Y) and rotate it
about the face normal to seat it optimally.

Units: millimetres. Binary or ASCII STL both work.

If your elements use a different convention, tell the app author and the
placement transform can be adjusted.
