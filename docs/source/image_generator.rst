.. _image_generator_module:

===============
Image Generator
===============

Overview
--------

.. versionadded:: 2.4

The image generator module allows users to generate simulated radiation
images. This module is different than other modules in that the goal here is non-deterministic. There are no phantom
analysis routines here. What is here started as a testing concept for pylinac itself, but has uses for advanced users
of pylinac who wish to build their own tools.

The module allows users to create a pipeline ala keras, where layers are added to an empty image. The user can add as
many layers as they wish.

Quick Start
-----------

The basics to get started are to import the image simulators and layers from pylinac and add the layers as desired.

.. plot::

    from matplotlib import pyplot as plt

    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()  # this will set the pixel size and shape automatically
    as1200.add_layer(FilteredFieldLayer(field_size_mm=(50, 50)))  # create a 50x50mm square field
    as1200.add_layer(GaussianFilterLayer(sigma_mm=2))  # add an image-wide gaussian to simulate penumbra/scatter
    as1200.generate_dicom(file_out_name="my_AS1200.dcm", gantry_angle=45)  # create a DICOM file with the simulated image
    # plot the generated image
    plt.imshow(as1200.image)

Layers & Simulators
-------------------

Layers are very simple structures. They usually have constructor arguments specific to the layer and always define an
``apply`` method with the signature ``.apply(image, pixel_size) -> image``. The apply method returns the modified image
(a numpy array). That's it!

Simulators are also simple and define the parameters of the image to which layers are added. They have ``pixel_size``
and ``shape`` properties and always have an ``add_layer`` method with the signature ``.add_layer(layer: Layer)``. They
also have a ``generate_dicom`` method for dumping the image along with mostly stock metadata to DICOM.

Extending Layers & Simulators
-----------------------------

This module is meant to be extensible. That's why the structures are defined so simply. To create a custom simulator,
inherit from ``Simulator`` and define the pixel size and shape:

.. code-block:: python

    from pylinac.core.image_generator.simulators import Simulator


    class AS5000(Simulator):
        pixel_size = 0.12
        shape = (5000, 5000)


    # use like any other simulator

To implement a custom layer, inherit from ``Layer`` and implement the ``apply`` method:

.. code-block:: python

    from pylinac.core.image_generator.layers import Layer


    class MyAwesomeLayer(Layer):
        def apply(image, pixel_size):
            # do stuff here
            return image


    # use
    from pylinac.core.image_generator import AS1200Image

    as1200 = AS1200Image()
    as1200.add_layer(MyAwesomeLayer())
    ...

Exporting Images to DICOM
-------------------------

The ``Simulator`` class has two methods for generating DICOM. One returns a Dataset and another fully saves it out to a file.

.. code-block:: python

    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()
    as1200.add_layer(FilteredFieldLayer(field_size_mm=(50, 50)))
    as1200.add_layer(GaussianFilterLayer(sigma_mm=2))

    # generate a pydicom Dataset
    ds = as1200.as_dicom(gantry_angle=45)
    # do something with that dataset as needed
    ds.PatientID = "12345"

    # or save it out to a file
    as1200.generate_dicom(file_out_name="my_AS1200.dcm", gantry_angle=45)

Examples
--------

Let's make some images!

Simple Open Field
^^^^^^^^^^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt
    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()  # this will set the pixel size and shape automatically
    as1200.add_layer(FilteredFieldLayer(field_size_mm=(150, 150)))  # create a 50x50mm square field
    as1200.add_layer(GaussianFilterLayer(sigma_mm=2))  # add an image-wide gaussian to simulate penumbra/scatter
    # plot the generated image
    plt.imshow(as1200.image)

Off-center Open Field
^^^^^^^^^^^^^^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt
    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()  # this will set the pixel size and shape automatically
    as1200.add_layer(FilteredFieldLayer(field_size_mm=(30, 30), cax_offset_mm=(20, 40)))
    as1200.add_layer(GaussianFilterLayer(sigma_mm=3))
    # plot the generated image
    plt.imshow(as1200.image)

Winston-Lutz FFF Cone Field with Noise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt
    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilterFreeConeLayer, GaussianFilterLayer, PerfectBBLayer, RandomNoiseLayer

    as1200 = AS1200Image()
    as1200.add_layer(FilterFreeConeLayer(50))
    as1200.add_layer(PerfectBBLayer(bb_size_mm=5))
    as1200.add_layer(GaussianFilterLayer(sigma_mm=2))
    as1200.add_layer(RandomNoiseLayer(sigma=0.02))
    # plot the generated image
    plt.imshow(as1200.image)

VMAT DRMLC
^^^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt
    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()
    as1200.add_layer(FilteredFieldLayer((150, 20), cax_offset_mm=(0, -40)))
    as1200.add_layer(FilteredFieldLayer((150, 20), cax_offset_mm=(0, -10)))
    as1200.add_layer(FilteredFieldLayer((150, 20), cax_offset_mm=(0, 20)))
    as1200.add_layer(FilteredFieldLayer((150, 20), cax_offset_mm=(0, 50)))
    as1200.add_layer(GaussianFilterLayer())
    plt.imshow(as1200.image)
    plt.show()

Picket Fence
^^^^^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt
    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()
    height = 350
    width = 4
    offsets = range(-100, 100, 20)
    for offset in offsets:
        as1200.add_layer(FilteredFieldLayer((height, width), cax_offset_mm=(0, offset)))
    as1200.add_layer(GaussianFilterLayer())
    plt.imshow(as1200.image)
    plt.show()

Starshot
^^^^^^^^

Simulating a starshot requires a small trick as angled fields cannot be handled by default. The following example
rotates the image after every layer is applied.

.. note:: Rotating the image like this is a convenient trick but note that it will rotate the entire existing image
          including all previous layers. This will also possibly erroneously adjust the horn effect simulation.
          Use with caution.

.. plot::

    from scipy import ndimage
    from matplotlib import pyplot as plt
    from pylinac.core.image_generator import AS1200Image
    from pylinac.core.image_generator.layers import FilteredFieldLayer, GaussianFilterLayer

    as1200 = AS1200Image()
    for _ in range(6):
        as1200.add_layer(FilteredFieldLayer((250, 7), alpha=0.5))
        as1200.image = ndimage.rotate(as1200.image, 30, reshape=False, mode='nearest')
    as1200.add_layer(GaussianFilterLayer())
    plt.imshow(as1200.image)
    plt.show()

Helper utilities
^^^^^^^^^^^^^^^^

Using the new utility functions of v2.5+ we can construct full dicom files of picket fence and winston-lutz sets of images:

.. code-block:: python

    from pylinac.core.image_generator import generate_picketfence, generate_winstonlutz
    from pylinac.core import image_generator

    sim = image_generator.simulators.AS1200Image()
    field_layer = image_generator.layers.FilteredFieldLayer  # could also do FilterFreeLayer
    generate_picketfence(
        simulator=Simulator,
        field_layer=FilteredFieldLayer,
        file_out="pf_image.dcm",
        pickets=11,
        picket_spacing_mm=20,
        picket_width_mm=2,
        picket_height_mm=300,
        gantry_angle=0,
    )
    # we now have a pf image saved as 'pf_image.dcm'

    # create a set of WL images
    # this will create 4 images (via image_axes len) with an offset of 3mm to the left
    # the function is smart enough to correct for the offset w/r/t gantry angle.
    generate_winstonlutz(
        simulator=sim,
        field_layer=field_layer,
        final_layers=[GaussianFilterLayer()],
        gantry_tilt=0,
        dir_out="./wl_dir",
        offset_mm_left=3,
        image_axes=[[0, 0, 0], [180, 0, 0], [90, 0, 0], [270, 0, 0]],
    )


Tips & Tricks
-------------

* The ``FilteredFieldLayer`` and ``FilterFree<Field, Cone>Layer`` have gaussian filters applied to create a first-order
  approximation of the horn(s) of the beam. It doesn't claim to be super-accurate, it's just to give some reality to the images.
  You can adjust the magnitude of these parameters to simulate other energies (e.g. sharper horns) when defining the layer.

* The ``Perfect...Layer`` s do not apply any energy correction as above.

* Use ``alpha`` to adjust the intensity of the layer. E.g. the BB layer has a default alpha of -0.5 to simulate attenuation.
  This will subtract out up to half of the possible dose range existing on the image thus far
  (e.g. an open image of alpha 1.0 will be reduced to 0.5 after a BB is layered with alpha=-0.5). If you want to simulate a thick material like
  tungsten you can adjust the alpha to be lower (more attenuation). An alpha of 1 means full radiation, no attenuation (like an open field).

* Generally speaking, don't apply more than one ``GaussianFilterLayer`` since they are additive. A good rule is to apply one filter at the
  end of your layering.

* Apply ``ConstantLayer`` s at the beginning rather than the end.

.. warning:: Pylinac uses unsigned int16 datatypes (native EPID dtype). To keep images from flipping bits when adding layers,
             pylinac will clip the values. Just be careful when, e.g. adding a ``ConstantLayer`` at the end of a layering. Better to do this
             at the beginning.

API Documentation
-----------------

Layers
^^^^^^

.. autoclass:: pylinac.core.image_generator.layers.PerfectConeLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.FilterFreeConeLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.PerfectFieldLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.FilteredFieldLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.FilterFreeFieldLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.PerfectBBLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.GaussianFilterLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.RandomNoiseLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.ConstantLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.SlopeLayer
        :inherited-members:

.. autoclass:: pylinac.core.image_generator.layers.ArrayLayer
        :inherited-members:

Simulators
^^^^^^^^^^

.. autoclass:: pylinac.core.image_generator.simulators.AS500Image
    :inherited-members:

.. autoclass:: pylinac.core.image_generator.simulators.AS1000Image
    :inherited-members:

.. autoclass:: pylinac.core.image_generator.simulators.AS1200Image
    :inherited-members:

Helpers
^^^^^^^

.. autofunction:: pylinac.core.image_generator.utils.generate_picketfence

.. autofunction:: pylinac.core.image_generator.utils.generate_winstonlutz

.. autofunction:: pylinac.core.image_generator.utils.generate_winstonlutz_cone

.. autofunction:: pylinac.core.image_generator.utils.generate_winstonlutz_multi_bb_multi_field
