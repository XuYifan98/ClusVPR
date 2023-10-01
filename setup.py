from setuptools import setup, find_packages


setup(name='ClusVPR',
      version='1.0',
      description='Open-source toolbox for Visual Place Recognition',
      author_email='Anonymous',
      url='https://github.com/aaai-2024/ClusVPR',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision', 'opencv-python',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
        'Viusal Place Recognition',
        'Vision Transformer',
        'Self-supervised'
      ])
