`Oberwolfach Seminar: Mathematics of Deep Learning <https://www.mfo.de/occasion/1842b>`_
========================================================================================
*14 October - 20 October 2018*

**Intended for participants of the seminar**

.. contents:: Navigation

About this repository
---------------------

Here you find all the information and materials needed to follow the **practical session** of the seminar. During this session we will give a short introduction to the deep learning framework **Tensorflow** and use it to numerically validate and explore some results concerning the approximation properties of deep ReLU neural networks. The content of the session is heavily based on the works of Dmitry Yarotsky. Have a look at his paper on `Error bounds for approximations with deep ReLU networks <https://www.sciencedirect.com/science/article/pii/S0893608017301545>`_ for more details. 

Content of this repository
--------------------------

TO BE ADDED SHORTLY


Requirements
------------

The session will provide possibilities to **interactively** experiment with some of the code snippets we provide here. If you want to actively participate you will need to have the software packages listed below **installed** on your system. The next section gives instructions about how to setup all the required packages using a **miniconda** virtual environment that you can (if you want to) easily remove from your system after the seminar. 

If you already have the required software installed, or you want to use a different installation method and know what you are doing: excellent. If not, simply follow the instruction of the next section.

Here is a list of what you will need:

- Python 3
- Tensorflow 
- Jupyter / IPython
- Scipy / Numpy
- Matplotlib / Pyplot


Installation instructions
-------------------------

1. Make sure you have a version of **Python 3** installed. You can find instructions on how to do it in the `Python wiki <https://wiki.python.org/moin/BeginnersGuide/Download>`_.

2. Install the **miniconda** package and environment manager. You can find instructions on how to do it in the `Conda documentation <https://conda.io/docs/user-guide/install/index.html>`_. Make sure to get the version for Python 3.

3. Create a new conda environment by running the following in your command line (you can choose a different name for the environment if you want):

    .. code-block:: bash
        
        $ conda create --name dl_seminar python=3

4. Activate the newly created environment by running

    .. code-block:: bash
        
        $ conda activate dl_seminar

   (Try :code:`$ source activate dl_seminar` instead of :code:`$ conda activate dl_seminar` if this did not work.)

5. Install all the required Python packages within the new environment by running

    .. code-block:: bash
        
        $ conda install scipy
        $ conda install matplotlib
        $ conda install jupyter
        $ conda install tensorflow

6. To test if your Tensorflow installation was successful you can open Python 3 in a command line and run

    .. code-block:: python

        >>> import tensorflow as tf
        >>> print(tf.__version__)
        >>> session = tf.Session()

   If the Tensorflow version is printed correctly and creating the Tensorflow session prints out some additional version information but does not throw an error then you have sucessfully installed Tensorflow and are ready for the practical session.
