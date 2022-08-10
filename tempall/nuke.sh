#!/bin/bash

deactivate
rm -rf ~/openfltest
virtualenv --python==3.8 ~/openfltest
source ~/openfltest/bin/activate
pip install openfl==1.3 tensorflow==2.9.1 protobuf==3.19tensorflow_datasets==4.6.0 jax==0.3.15 jaxlib==0.3.15+cuda11.cudnn82  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
