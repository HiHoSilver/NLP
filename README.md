## Testing NLP for Extracting Research Abstract Topics

Testing Natural Language Processing with Latent Dirichlet Allocation (LDA), pseudo-document splitting, and bigram and trigram models to extract topics from research project and paper abstracts.

## Requirements

Install the required libraries:

	pip install -r requirements.txt

Install the required `NLTK` resources

    nltk.downloader punkt_tab

If you plan to use Jupyter Notebook and are using a virutal environment, you will also need to install `ipykernel` and register the virtual environment as a Jupyter Kernel:

    pip install ipkernel
    ipykernel install --user --name=[env_name]