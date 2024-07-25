Troubleshooting
---------------

`malloc` error querying database
================================

If you get a `malloc` error when querying the database, this might arise from `psycopg2` (the postgres database driver) and `openssl` being incompatible. `openssl` version 3.1.4 is compatible with the version of `psycopg2` used

`llama-cpp-python` build errors
===============================

Issues have been encountered when installing `llama-cpp-python` on some linux distributions. If it fails to build the `llama.cpp` wheel, it might be failing to find your `openmp` library. There are a couple of solutions.

First solution
##############

The first is to install `llama.cpp` without OpenMP. Refer to the `llama-cpp-python` documentation. This can have performance issues as it will be single threaded

Second solution
###############
The second is to locate your `fopenmp` library

.. code-block:: console
            
    gcc -fopenmp -dM -E - < /dev/null | grep -i openmp
    find /usr/lib64 -name "libgomp.so*"
        
Then set your library paths to include these directories

.. code-block:: console
            
    export LD_LIBRARY_PATH=/path/to/directory/containing/libgomp.so:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/path/to/directory/containing/libgomp.so:$LIBRARY_PATH
         
Install `llama-cpp-python` with these set explicitly

.. code-block:: console
            
    CFLAGS="-fopenmp" CXXFLAGS="-fopenmp" pip install llama-cpp-python --no-cache-dir

Good luck!
          
