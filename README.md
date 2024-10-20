# MARVEL: Microenvironment Annotation by supeRVised Graph ContrastivE Learning

## Description

Recent advancements in *in situ* molecular profiling technologies, including spatial proteomics and transcriptomics, have enabled detailed characterization of the microenvironment at cellular and subcellular levels. While these techniques provide rich information about individual cells' spatial coordinates and expression profiles, how to effectively identify biologically meaningful spatial structures remains a significant challenge. Current methodologies rely on aggregating features from neighboring cells for spatial environment annotation, which is labor-intensive and demands extensive domain knowledge.

To address these challenges, we propose a supervised graph contrastive learning framework: **Microenvironment Annotation by supeRVised Graph ContrastivE Learning (MARVEL)**. This framework combines supervised contrastive learning with graph contrastive learning methods to effectively map local microenvironments, represented by cell neighbor graphs, into a continuous representation space, facilitating various downstream microenvironment annotation scenarios. By leveraging partially annotated samples as strong positives, our approach mitigates the common issues of false positives encountered in conventional graph contrastive learning.

Using real-world annotated data, we demonstrate that MARVEL outperforms existing methods in three key microenvironment-related tasks:
- Transductive microenvironment annotation
- Inductive microenvironment querying
- Identification of novel microenvironments across different tissue slices.

## Environment Setup

To set up the environment for MARVEL, please follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/C0nc/marvel.git
   cd marvel
   ```

2. Create a new virtual environment (optional but recommended):

   ```bash
   python -m venv marvel-env
   source marvel-env/bin/activate  # On Windows use `marvel-env\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that you have the correct data files in the `data/` directory. If not, please download the dataset from [link-to-dataset].

## Running MARVEL

To run the MARVEL framework, follow the steps below:

1. Preprocess the data (if necessary):

   ```bash
   python preprocess_data.py --input data/raw --output data/processed
   ```

2. Train the MARVEL model:

   ```bash
   python train.py --config configs/config.yaml
   ```

3. Run downstream tasks, such as microenvironment annotation:

   ```bash
   python annotate.py --input data/processed --model outputs/marvel_model.pt --task transductive
   ```

4. Evaluate the results:

   ```bash
   python evaluate.py --predictions results/predictions.csv --groundtruth data/groundtruth.csv
   ```

## Citation


