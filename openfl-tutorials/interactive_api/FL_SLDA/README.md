# Federated Learning - SLDA Algorithm (CIFAR-10 dataset)

## I. About the Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
<br/>

## II. About the Federation

Data is equally partitioned between envoys/participants.

You can write your own splitting schema in the shard descriptor class.
<br/>

## III. How to run this tutorial (without TLS and locally as a simulation):

### 0. If you haven't done so already, create a virtual environment and install OpenFL:

  ```sh
  pip install openfl==1.4
  ```

  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/install.html#install-the-package).
<br/>
 
### 1. Split terminal into 3 (1 terminal for the director, 1 for the envoy, and 1 for the experiment)
<br/> 

### 2. Do the following in each terminal:
   - Activate the virtual environment from step 0:
   
   ```sh
   source <virtualenv>/bin/activate
   ```
   - If you are in a network environment with a proxy, ensure proxy environment variables are set in each of your terminals.
   - Navigate to the tutorial:
    
   ```sh
   cd examples/federated_learning/slda/
   ```
<br/>

### 3. In the first terminal, run the director:

```sh
cd director
./start_director.sh
```
<br/>

### 4. In the second terminal run the envoy:

```sh
cd envoy
./start_envoy.sh env_one envoy_config_one.yaml
```

Optional: Run a second envoy in an additional terminal:
  - Ensure step 2 is complete for this terminal as well.
  - Run the second envoy:
```sh
cd envoy
./start_envoy.sh env_two envoy_config_two.yaml
```
<br/>

### 5. In the third terminal (or forth terminal, if you chose to do two envoys) run the Jupyter Notebook:

```sh
cd workspace
jupyter notebook FCL_SLDA_Tutorial_CIFAR-10.ipynb
```
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment was finished successfully.