from setuptools import setup

#never tested :)

setup(
   name='NewsAnalysis',
   version='1.0',
   description='A module which provides various NLP tools for news analysis',
   author='Andrei Simion',
   author_email='',
   packages=['NewsAnalysis'],  #same as name
   install_requires=['nltk', 'greek'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)