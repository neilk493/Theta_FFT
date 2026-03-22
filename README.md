Theta_FFT: A Modular Pipeline for FFT-Based Analysis of Theta-pp Protein Backbone Geometry


--- Overview ---

Theta_FFT is a computational biology research pipeline developed at the Williams Lab, Georgia Institute of Technology. It applies Fast Fourier Transform analysis to the theta-pp geometric descriptor of protein backbone structure, with the goal of identifying and comparing local structural motifs across proteins without relying on sequence alignment or conventional secondary structure assignments.


--- What is Theta-pp: ---

Theta-pp is a signed dihedral angle measured between the normal vectors of two consecutive peptide planes along a protein backbone. For each pair of adjacent residues, it produces a single scalar value in degrees. When computed across an entire chain, the result is a one-dimensional signal indexed by residue position called the theta fingerprint.

Unlike phi and psi torsion angles, which describe individual bond rotations independently, theta-pp captures how peptide planes propagate relative to each other along the chain. This makes it sensitive to intermediate-scale geometric organization that DSSP annotations and Ramachandran statistics miss.

Known properties of the theta signal:

Alpha helices produce smooth, low-variance fingerprints with values clustered around positive 90 to 100 degrees.

Beta strands produce fingerprints clustered around negative 100 to 140 degrees with characteristic residue-to-residue alternation.

Loop regions produce high-variance, scattered values with no dominant regime.


--- Why FFT: ---

Because the theta fingerprint is a one-dimensional spatial signal indexed by residue position, it can be analyzed using FFT as a feature extraction method. Low frequencies in the spectrum correspond to slow, large-scale geometric changes along the chain such as fold-level organization. Higher frequencies correspond to rapid local oscillations characteristic of secondary structure patterns.

The central hypothesis of this pipeline is that sliding-window FFT applied to the theta fingerprint produces position-resolved local spectral signatures that are characteristic of structural class, and that these signatures are consistent enough across different proteins to enable motif comparison without sequence alignment.


--- Pipeline Structure: ---

The pipeline is fully modular. Each stage is a separate script. Analysis logic, plotting logic, and preprocessing logic are never mixed. Every module writes its outputs as CSV or JSON files that serve as inputs to the next stage.

Module 1: Global Spectral Analysis

Script: global_spectral_analysis.py
Input: fft_data/
Output: output/global_spectra/

Performs global FFT on each contiguous observed segment of each protein chain. Computes per-segment power spectra, dominant frequencies, band power features, and spectral summary statistics. Produces a combined long-format spectrum table, a segment-level feature table, a skipped segment log, and run metadata. No plots are generated here.

Module 2: Local Sliding-Window Spectral Analysis

Script: local_spectral_analysis.py
Input: fft_data/
Output: output/spectrograms/

Slides a fixed-size window along each contiguous segment and computes FFT at every position. Default window size is 16 residues with step size 1. Produces a per-window feature table with 34 columns including theta statistics, spectral features, band power fractions, and autocorrelation at lags 1 through 10. Also produces per-segment local spectra tables and a combined long-format table. No plots are generated here.

Module 3: Protein and Motif Comparison

Script: protein_motif_comparison.py
Input: output/spectrograms/local_window_features.csv
Output: output/comparison/

Standardizes the window-level feature vectors, computes pairwise Euclidean distances across all windows in standardized feature space, identifies cross-protein nearest neighbors, extracts reciprocal best-match motif candidate pairs, builds a protein-to-protein similarity matrix using mean minimum distances, and runs unsupervised k-means clustering with k-means++ initialization implemented in pure numpy to assign windows to motif families. No plots are generated here.

Module 4: Validation Against Known Structure Labels

Script: validation.py
Input: output/comparison/, pdb/
Output: output/validation/

Maps DSSP secondary structure assignments from PDB files onto the window-level feature table by residue position. Evaluates motif cluster purity against DSSP labels. Computes precision and recall for each structural class using a corrected DSSP simplification where H covers codes H, G, I, and P; E covers codes E and B; and L covers everything else. Requires mkdssp to be installed and accessible on the system PATH.

Plotting Scripts:

plot_global_spectra.py reads output/global_spectra/global_spectra_long.csv and produces per-protein power spectrum line plots saved to output/global_spectra_plots/

plot_local_spectrograms.py reads output/spectrograms/local_spectra_long.csv and produces per-segment spectrogram heatmaps with frequency on the y-axis and residue position on the x-axis saved to output/spectrogram_plots/

plot_comparison.py reads all outputs from output/comparison/ and produces five publication-quality figures saved to output/comparison_plots/ including a 2D and 3D t-SNE embedding of all windows colored by protein and by motif cluster, a protein similarity heatmap with structural class banding, a spectral distance distribution plot by structural class pairing, and a motif cluster protein composition chart

plot_validation.py reads all outputs from output/validation/ and produces eight validation figures saved to output/validation_plots/ including a cluster purity heatmap, a t-SNE recolored by DSSP majority label, per-class precision and recall bars, per-protein DSSP composition and accuracy, and supporting ambiguity summaries

plot_tsne_3d.py generates 3D t-SNE embeddings colored by protein and by motif cluster, with alternate camera angle views, and exports the 3D coordinates for downstream use

export_tsne_3d_for_desmos.py exports the 3D t-SNE coordinates in grouped formats organized by motif cluster and by protein for interactive exploration in Desmos 3D


--- Interactive 3D Motif Map: ---

The 3D t-SNE embedding of all 1123 windows has been exported to Desmos 3D as an interactive, rotatable point cloud organized by motif cluster and by protein. This allows the motif-space geometry to be explored spatially beyond what static matplotlib figures can show.

The live interactive map is available here: [https://www.desmos.com/3d/mbs7ixsydd](url)

The exported coordinate files are stored in output/desmos_3d_exports/ organized by motif cluster and by protein, with import instructions and color guides included in output/desmos_3d_exports/manifests/


--- Directory Structure: ---

Theta_FFT/

fft_data/ contains preprocessed theta-pp CSV files for each protein named XXXX_fft_data.csv

pdb/ contains raw PDB structure files for each protein

raw_theta/ contains upstream geometry computation outputs including normal vectors and adjacent angle CSVs

archive/ contains earlier experimental scripts retained for reference

output/ contains all pipeline outputs organized by module

output/global_spectra/ outputs from Module 1

output/spectrograms/ outputs from Module 2

output/comparison/ outputs from Module 3

output/validation/ outputs from Module 4

output/global_spectra_plots/ plots from plot_global_spectra.py

output/spectrogram_plots/ plots from plot_local_spectrograms.py

output/comparison_plots/ plots from plot_comparison.py

output/validation_plots/ plots from plot_validation.py

output/desmos_3d_exports/ 3D t-SNE coordinate exports organized by motif cluster and by protein


--- Proteins in the Dataset: ---

1AL1 is a pure alpha helix used as a geometric baseline

1TEN is a pure beta sheet, a fibronectin type III domain with a 7-stranded beta sandwich

1UBQ is mixed alpha plus beta, ubiquitin with a central helix and five-stranded sheet

1PKK is mixed alternating, a TIM barrel with 8 repeating beta-alpha units

1GZM is helix-dominated with 7 transmembrane helices

2HHB is helix-dominated, human hemoglobin in the globin fold approximately 75 percent helical

1FNA is a pure beta sheet, a fibronectin type III module used for cross-protein sheet comparison

2IGF is a pure beta sheet, an immunoglobulin Fab fragment with characteristic Ig folds

1LYZ is mixed, hen egg-white lysozyme with multiple helices and a small antiparallel sheet

2PTN is beta-sheet-dominated, trypsin composed of two beta barrel domains


--- Key Results: ---

The t-SNE embedding of 1123 windows from 9 proteins shows clear spatial separation of helix-dominated windows from sheet and mixed windows with no structural labels used during computation. The two helix proteins form a compact isolated cluster in the embedding, visible in both 2D and 3D projections.

The protein similarity matrix shows the highest off-diagonal similarity between 1TEN and 1FNA at 0.360, which are two structurally equivalent proteins sharing the fibronectin type III fold. The helix proteins 2HHB and 1GZM show elevated mutual similarity at 0.350. The lowest values in the matrix are helix-to-sheet cross-class comparisons.

Unsupervised k-means clustering with 6 clusters spontaneously produced two helix-enriched clusters without any structural labels as input. Motif003 is 60.8 percent 1GZM and 34.7 percent 2HHB. Motif004 is 73.9 percent 1GZM and 20.3 percent 2HHB.

The spectral distance analysis shows a monotonic increase from helix-helix mean distance 1.84 to sheet-sheet mean 2.07 to mixed-mixed mean 2.41 to cross-class mean 2.79, consistent with the hypothesis that same-class windows are more spectrally similar than cross-class windows.

166 reciprocal best-match cross-protein window pairs were identified including a series of high-similarity matches between 1TEN and 1FNA windows with similarity scores above 0.70, representing candidate structurally equivalent local regions in two proteins sharing the same fold.

DSSP validation confirms that motif003 and motif004 are essentially pure helix clusters with purity scores of 1.000 and 0.978 respectively. Class-level validation yields helix precision of 0.838 and recall of 0.767 with F1 of 0.801. Sheet is partially recovered through motif000 which emerges as weakly sheet-majority. Loop remains diffuse across the remaining mixed clusters.


--- Signal Handling Conventions: ---

The signal is theta_signed in degrees as stored in the fft_data CSVs. The x-axis is seq_index, a sequential zero-based integer index used consistently across all modules and never replaced with PDB residue numbers. Gaps in PDB residue numbering are handled by splitting chains into contiguous segments at rows where has_gap_before equals 1. Windows never cross gaps. No smoothing, interpolation, unwrapping, or angle transformation is applied at any stage. Mean centering is applied per window before FFT.


--- Dependencies: ---

Python 3.10 or higher, numpy, pandas, matplotlib, biopython

mkdssp must be installed separately for Module 4 validation. On conda-based systems install with: conda install -c conda-forge dssp

All analysis modules use only Python standard library, numpy, and pandas. No scipy, no sklearn, no external FFT libraries. The t-SNE implementation in plot_comparison.py and plot_validation.py is written from scratch in numpy.


--- How to Run: ---

Run modules in order from the project root directory.

python global_spectral_analysis.py

python local_spectral_analysis.py

python protein_motif_comparison.py

python validation.py --verbose

python plot_global_spectra.py

python plot_local_spectrograms.py

python plot_comparison.py

python plot_validation.py

Note: validation.py must be run from an environment where mkdssp is accessible on PATH. On Windows with conda this means running from Anaconda Prompt rather than standard PowerShell.

All scripts accept command-line arguments. Run any script with --help to see available options including input and output directory overrides, window size, step size, number of clusters, t-SNE parameters, and verbosity.


--- Affiliation: ---

Williams Lab, School of Chemistry and Biochemistry, Georgia Institute of Technology.
