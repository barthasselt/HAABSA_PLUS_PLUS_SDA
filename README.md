# HAABSA_PLUS_PLUS_SDA

Notes:
- For the ontology reasoner, the original code of Olaf Wallaart from https://github.com/ofwallaart/HAABSA.git is used. 
- For the LCR-rot-hop++, an adjusted version of the code of Wessel van Ree from https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus.git is used. The only differences are that it specifies the seeds more strictly and the code is converted to work on stand-alone base, without any CLI required. Only the code files that have been changed are included in the folder.
- The data augmentation code is partially build on the code of Bron Hollander from https://github.com/BronHol/HAABSA_PLUS_PLUS_DA.git.

Running the code for data augmentation: 
- Start by installing the packages from the requirements.txt file.
- Continue by finetuning the transformer you want to use. For each transformer there is a specific code that you can run. You need to specify the correct paths to the source files and output files in order for it to work.
- After finetuning, the model can be used for augmentation. For this purpose, each transformer (except BERTprepend and BERTexpand, that are joint together), also has its own code. You can select the word selection methods employed in the main function.
- The word selection functions are stored in a separate code, called: maskSelectionMethods.


