# TBGHofstadter
Code for calculating Hofstadter diagrams in the Bistrizer-Macdonald model of twisted bilayer graphene. I used it to generate the diagram in https://arxiv.org/abs/2106.10650.

The implementation is in the continuum model and follows that of Bistrizer &amp; Macdonald in PhysRevB.84.035440. That is, the Hamiltonian is defined in the basis of Landau levels with the same periodicity as the Moire structure.

# Basic Usage:
The model is contained in the LandauDiracModel object. An example for defining model parameters and generating the figure is provided in the main file.
