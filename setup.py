from distutils.core import setup
setup(
  name = 'GMMchisquare',         # How you named your package folder (MyLib)
  packages = ['GMMchisquare'],   # Chose the same as "name"
  version = '0.22',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'GMM with chi-square protocol',   # Give a short description about your library
  author = 'Ta-Chun (Jeff) Liu',                   # Type in your name
  author_email = 'ta-chun.liu@oncology.ox.ac.uk',      # Type in your E-Mail
  url = 'https://github.com/jeffliu6068/GMMchisquare',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/jeffliu6068/GMMchisquare/archive/v_02.tar.gz',    # I explain this later on
  keywords = ['GMM', 'Chi-square', 'Categorization'],   # Keywords that define your package best
  install_requires=[            
          'pandas',
          'scipy',
          'numpy',
          'matplotlib',
          'seaborn',
          'sklearn',
          'tqdm',
      ],
  
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Public',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
