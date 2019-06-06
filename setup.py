from setuptools import setup, find_packages

setup(name="squad",
      version="0.1",
      packages=find_packages(),
      description="Example",
      author="Nicky",
      author_email="nikolay.karagyozov1@gmail.com",
      license="Default",
      install_requires=[
        'keras',
        'h5py'
      ],
      zip_safe=False)
