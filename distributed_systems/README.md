This will hold progress and documentation of integration of applications in the frontier framework as well as the physical setup used to leverage the the framework.



Current work done:
- understanding the maven structure of the whole frontier project


#### =====

I've added a nice little bash script to run things faster under `scripts`:

#### Usage:

- Load up the `run_type.sh` (make sure you `sudo chmod 777` this one) and `script.py` files inside the frontier source directory.
- Usage:
  - `./run_type.sh build <master_ip_address>` - the user chooses what ip address to insert for the master.
  - `./run_type.sh master>` - this will run the node as the master
  - `./run_type.sh worker>` - this will run the node as a worker

#### =====
