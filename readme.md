pref_voting
==========

## Installation

With pip package manager:

```bash
pip install pref_voting
```


## Profiles and Voting Methods

A profile (of linear orders over the candidates) is created by initializing a Profile class object.  This needs a list of rankings (each ranking is a tuple of numbers), the number of candidates, and a list giving the number of each ranking in the profile:

```python
from pref_voting.profiles import Profile

rankings = [(0, 1, 2, 3), (2, 3, 1, 0), (3, 1, 2, 0), (1, 2, 0, 3), (1, 3, 2, 0)]
num_cands = 4
rcounts = [5, 3, 2, 4, 3]

prof = Profile(rankings, num_cands, rcounts=rcounts)
```

The function generate_profile is used to generate a profile for a given number of candidates and voters:  
```python
from pref_voting.generate_profiles import generate_profile

# generate a profile using the Impartial Culture probability model
prof = generate_profile(3, 4) # prof is a Profile object with 3 candidate and 4 voters

# generate a profile using the Impartial Anonymous Culture probability model
prof = generate_profile(3, 4, probmod = "IAC") # prof is a Profile object with 3 candidate and 4 voters
```

Voting methods  

```python
from pref_voting.profiles import Profile
from pref_voting.voting_methods import *

prof = Profile(rankings, num_cands, rcounts=rcounts)
print(f"The {split_cycle_faster.name} winners are {split_cycle_faster(prof)}")
```

## Versions

- v0.1.0 (2022-07-08): **Initial release** 

## Questions?

Feel free to [send me an email](https://pacuit.org/) if you have questions about the project.

## License

[MIT](https://github.com/jontingvold/pyrankvote/blob/master/LICENSE.txt)