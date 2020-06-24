# metaspace_msms_paper
  
METASPACE MSMS is a workflow for assigning in-source fragments in mass spectrometry imaging datasets.

This workflow is described and examples are shown to support the draft manuscript X.

I. Workflow steps on spotted standards include:

    1. Authentic standard compounds of interest (COI's).
    
    2. Downloading database references.
    
    3. Collecting predicted MS/MS spectra for COI's
    
    4. Collecting reference experimental MS/MS spectra for COI's.
    
    5. Collecting in-house experimental MS/MS spectra for COI's.
    
    6. Generating custom MS1 database using known spotted compounds.
    
    7. Running and interpreting the METASPACE results.
    
    8. Generating custom MS2 database using known spotted compounds.
    
    9. Running and interpreting the METASPACE MSMS results.
    
    10. Generate psuedo-MS/MS spectra from ISF data.
    
    11. Compare predicted, experimental, and psuedo-MS/MS spectra.

    12. Pick winners and carry well-behaved molecules forward.

II. Workflow steps on real samples include:

    1. List of ds_ids and metadata for selected experiments.
    
    2. Download METASPACE results.
    
    3. Generating custom MS2 database using observed compounds.
    
    4. Running and interpreting the METASPACE MSMS results.
    
    5. Generate colocalization weighted psuedo-MS/MS spectra from ISF data.
    
    6. Score predicted, experimental, and psuedo-MS/MS spectra together for well-behaved subset.
    
    7. Plot examples.

III. Over-representation and tf-idf analysis.

    1. Apply to examples of interest, displaying isobars or isomers.
    
    2. Plot examples.

IV. Isomer resolution.

    1. Identify spotted and tissue well-behaved isomer sets within specific datasets.
    
    2. Illustrate examples.
