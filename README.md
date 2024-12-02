
Code for ShapeMI MICCAI paper - "MASSM: An End-to-End Deep Learning Framework for Multi-Anatomy Statistical Shape Modeling Directly From Images"


## Data 
Download the TotalSegmentator dataset from https://zenodo.org/records/10047292

ShapeWorks [(https://sciinstitute.github.io/ShapeWorks/latest/)](https://sciinstitute.github.io/ShapeWorks/latest/) was used for creating ground truth particles. Go through the linked website on how to install ShapeWorks and get the particles. Alternatively you can also have a look at ```util/ta_preprocess.py```, ```util/preprocess_for_shapeworks.py```, ```util/run_shapeworks_incremental.py``` and ```util/run_fd.py```.

After pre-processing save the paths in a json file, see ```util/data_loaders.py``` ```load_paths``` inside ```MultiClassDataset_FA``` for format.

## Training
```config.py``` for configuration

```
python train.py
```
## Evaluation
```
python test.py
```
