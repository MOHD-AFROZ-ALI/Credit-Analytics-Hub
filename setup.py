from setuptools import setup, find_packages

setup(
    name='credit_analytics_hub',
    version='0.1.0',
    description='Credit Analytics Hub - Model evaluation and training interface',
    author='MOHD AFROZ ALI',
    author_email='afrozali3001.aa@gmail.com',
    packages=find_packages(include=['pages', 'pages.*']),
    install_requires=[
        'streamlit'
        
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)